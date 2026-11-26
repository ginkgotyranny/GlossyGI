// NCR on HarmonyOS: the native side of the app.
//
//   ArkTS (EntryAbility, pages/Index)  ->  init(resourceManager, filesDir, args), setForeground,
//                                          getState() (JSON for the panel), set(name, value),
//                                          command(name, argument), cameraStick / cameraLift /
//                                          cameraLook (the page's touches on the picture)
//   XComponent                         ->  surface created / changed / destroyed (UI thread)
//   render thread                      ->  assets, Vulkan device, and per scene the NCR runtime
//                                          (ohos/replay); per frame: runtime.frame(), the Display
//                                          pass, a copy into the swapchain image, present
//
// The runtime is the one the command-line replayer uses: the frames of a Falcor bake, rendered and
// trained on the phone, the MLPs on the NPU. Everything the native side prints (stdout, stderr) goes
// to <files>/ncr.log and to hilog (tag NCR).
//
// Scenes are baked by ohos/tools/bake_scene.py and live on the phone as <files>/scenes/<scene>/<variant>
// (variant: static or anim; ohos/app/tools/push_scenes.sh puts them there) or, bundled in the HAP,
// as <files>/ncr/scenes/... The panel picks one; the render thread takes the current runtime down
// and brings the next one up. The device and the swapchain stay.
//
// Launch arguments (one string, --name=value or --flag): --scene=<name> and --variant=static|anim
// (default: the scene of the last run, else the first one found), --networks=npu|tiles,
// --model=bake|<checkpoint path>, --train-tile=<n>, --no-train, --copy-tiles, --cpu-prefix-sum,
// --loss=<every n frames>, --log-every=<n frames>, --set=<name>:<value>[,<name>:<value>...] (settings
// of the panel, see Runtime::settingNames), --dt=<seconds> (animation step per frame instead of the
// clock), --save-at=<frame>: after that frame write display.bin, trained.ckpt and schedule.txt to
// <files>/out (and with --save-results every image and ray-batch buffer, as the replayer does).
#include "ncr_runtime.h"

#include <ace/xcomponent/native_interface_xcomponent.h>
#include <hilog/log.h>
#include <napi/native_api.h>
#include <qos/qos.h>
#include <rawfile/raw_file.h>
#include <rawfile/raw_file_manager.h>

#include <dirent.h>
#include <ftw.h>
#include <pthread.h>
#include <sys/stat.h>
#include <unistd.h>

#include <condition_variable>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

using namespace replay;

namespace
{
// ------------------------------------------------------------------------------------------------
// State shared by the UI thread (NAPI calls, XComponent callbacks) and the render thread.
struct Shared
{
    std::mutex mutex;
    std::condition_variable changed;
    bool initialised = false;
    NativeResourceManager* resources = nullptr;
    std::string filesDir;
    std::string args;
    OHNativeWindow* window = nullptr;   // while the XComponent's surface exists
    bool surfaceGone = false;           // the UI thread asks the render thread to let go of the window
    bool windowClaimed = false;         // the render thread presents to the window
    bool foreground = true;
    bool threadStarted = false;
    std::string status = "starting";
    std::string state = "{\"status\":\"starting\"}";   // what the panel shows, as JSON

    // From the panel, taken by the render thread at the start of a frame.
    std::vector<std::pair<std::string, double>> sets;
    std::vector<std::pair<std::string, std::string>> commands;
    // The camera controls (pages/Index.ets): the stick (x right, y down, within the unit circle) and
    // the lift (-1 down, 1 up) as they are now, the look as radians turned since the last frame.
    double stickX = 0.0, stickY = 0.0, lift = 0.0, lookYaw = 0.0, lookPitch = 0.0;
};
Shared gShared;

void setStatus(const std::string& text)
{
    std::lock_guard<std::mutex> lock(gShared.mutex);
    gShared.status = text;
}

// ------------------------------------------------------------------------------------------------
// Logging: stdout and stderr go through a pipe to a thread that writes every line to the log file
// and to hilog. hilog alone drops lines (a rate-limited ring buffer); the file is complete.
void startLogging(const std::string& path)
{
    static bool started = false;
    if (started) return;
    started = true;
    int fds[2];
    if (pipe(fds) != 0) return;
    dup2(fds[1], STDOUT_FILENO);
    dup2(fds[1], STDERR_FILENO);
    close(fds[1]);
    setvbuf(stdout, nullptr, _IOLBF, 0);   // line-buffered: the last lines before a crash are not lost
    setvbuf(stderr, nullptr, _IONBF, 0);
    FILE* file = std::fopen(path.c_str(), "w");
    std::thread([fd = fds[0], file] {
        std::string line;
        char buffer[4096];
        for (;;)
        {
            const ssize_t n = read(fd, buffer, sizeof(buffer));
            if (n <= 0) break;
            for (ssize_t i = 0; i < n; i++)
            {
                if (buffer[i] != '\n')
                {
                    line += buffer[i];
                    continue;
                }
                if (file)
                {
                    std::fputs(line.c_str(), file);
                    std::fputc('\n', file);
                    std::fflush(file);
                }
                OH_LOG_Print(LOG_APP, LOG_INFO, 0x0000, "NCR", "%{public}s", line.c_str());
                line.clear();
            }
        }
    }).detach();
}

// ------------------------------------------------------------------------------------------------
// Assets: rawfile/ncr in the HAP (written by ohos/app/tools/sync_assets.py) is extracted to <files>/ncr
// on the first start and whenever its manifest changes; afterwards everything is read with plain file
// I/O, as on the desktop. manifest.txt: one line per file, "<sha1>\t<path relative to ncr/>".
bool readRawFile(NativeResourceManager* resources, const std::string& name, std::string& text)
{
    RawFile* raw = OH_ResourceManager_OpenRawFile(resources, name.c_str());
    if (!raw) return false;
    const long size = OH_ResourceManager_GetRawFileSize(raw);
    text.assign((size_t)std::max(size, 0L), '\0');
    const int read = size > 0 ? OH_ResourceManager_ReadRawFile(raw, &text[0], (size_t)size) : 0;
    OH_ResourceManager_CloseRawFile(raw);
    return read == size;
}

std::string readTextFile(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    std::ostringstream text;
    text << f.rdbuf();
    return text.str();
}

bool fileExists(const std::string& path)
{
    struct stat info;
    return stat(path.c_str(), &info) == 0;
}

void makeDirectories(const std::string& path)
{
    for (size_t slash = path.find('/', 1); slash != std::string::npos; slash = path.find('/', slash + 1)) mkdir(path.substr(0, slash).c_str(), 0770);
    mkdir(path.c_str(), 0770);
}

/// Deletes a folder and everything in it.
void removeTree(const std::string& path)
{
    nftw(
        path.c_str(), [](const char* file, const struct stat*, int, struct FTW*) { return std::remove(file) == 0 ? 0 : 0; }, 16,
        FTW_DEPTH | FTW_PHYS);
}

void extractAssets(NativeResourceManager* resources, const std::string& target)
{
    std::string manifest;
    if (!readRawFile(resources, "ncr/manifest.txt", manifest)) fail("asset extraction failed: rawfile/ncr/manifest.txt is missing (run ohos/app/tools/sync_assets.py)");
    const std::string installed = target + "/manifest.txt";
    if (readTextFile(installed) == manifest)
    {
        std::printf("assets up to date in %s\n", target.c_str());
        return;
    }
    // What an earlier version extracted goes first (an older HAP carried a scene), and the manifest
    // is written last, so an interrupted extraction is redone.
    removeTree(target);
    std::vector<std::string> files;
    std::istringstream lines(manifest);
    std::string line;
    while (std::getline(lines, line))
    {
        const size_t tab = line.find('\t');
        if (tab != std::string::npos) files.push_back(line.substr(tab + 1));
    }
    const auto start = std::chrono::steady_clock::now();
    std::vector<char> buffer(4 << 20);
    double bytes = 0.0;
    for (size_t i = 0; i < files.size(); i++)
    {
        setStatus(format("extracting assets %zu / %zu", i + 1, files.size()));
        const std::string& name = files[i];
        const std::string path = target + "/" + name;
        makeDirectories(path.substr(0, path.find_last_of('/')));
        RawFile* raw = OH_ResourceManager_OpenRawFile(resources, ("ncr/" + name).c_str());
        if (!raw) fail("asset extraction failed: rawfile/ncr/" + name + " is not in the HAP");
        FILE* out = std::fopen(path.c_str(), "wb");
        if (!out)
        {
            OH_ResourceManager_CloseRawFile(raw);
            fail("asset extraction failed: cannot write " + path);
        }
        for (;;)
        {
            const int n = OH_ResourceManager_ReadRawFile(raw, buffer.data(), buffer.size());
            if (n <= 0) break;
            std::fwrite(buffer.data(), 1, (size_t)n, out);
            bytes += n;
        }
        std::fclose(out);
        OH_ResourceManager_CloseRawFile(raw);
    }
    writeFile(installed, manifest.data(), manifest.size());
    std::printf("extracted %zu files, %.1f MB, to %s in %.1f s\n", files.size(), bytes / 1e6, target.c_str(), msSince(start) / 1000.0);
}

/// A baked scene on the phone: media/scenes/<scene> as its static or animated scene file.
struct SceneEntry
{
    std::string scene, variant, dir;
    std::string key() const { return scene + "/" + variant; }
};

std::vector<std::string> subdirectories(const std::string& path)
{
    std::vector<std::string> names;
    if (DIR* dir = opendir(path.c_str()))
    {
        while (dirent* entry = readdir(dir))
        {
            const std::string name = entry->d_name;
            struct stat info;
            if (name != "." && name != ".." && stat((path + "/" + name).c_str(), &info) == 0 && S_ISDIR(info.st_mode)) names.push_back(name);
        }
        closedir(dir);
    }
    std::sort(names.begin(), names.end());
    return names;
}

/// The packed bakes under <files>/scenes (pushed) and <files>/ncr/scenes (from the HAP):
/// <scene>/<variant>/replay.txt, or <scene>/replay.txt for a bake without a variant folder (static).
std::vector<SceneEntry> findScenes(const std::string& filesDir)
{
    std::vector<SceneEntry> scenes;
    auto add = [&](const std::string& scene, const std::string& variant, const std::string& dir) {
        for (const SceneEntry& e : scenes)
            if (e.scene == scene && e.variant == variant) return;
        scenes.push_back({scene, variant, dir});
    };
    for (const std::string& root : {filesDir + "/scenes", filesDir + "/ncr/scenes"})
    {
        for (const std::string& scene : subdirectories(root))
        {
            const std::string dir = root + "/" + scene;
            if (fileExists(dir + "/replay.txt")) add(scene, "static", dir);
            for (const std::string& variant : subdirectories(dir))
                if (fileExists(dir + "/" + variant + "/replay.txt")) add(scene, variant, dir + "/" + variant);
        }
    }
    std::sort(scenes.begin(), scenes.end(), [](const SceneEntry& a, const SceneEntry& b) {
        return a.scene != b.scene ? a.scene < b.scene : a.variant > b.variant;   // static before anim
    });
    return scenes;
}

std::map<std::string, std::string> parseArgs(const std::string& text)
{
    std::map<std::string, std::string> args;
    std::istringstream words(text);
    std::string word, pendingKey;
    while (words >> word)
    {
        if (word.rfind("--", 0) == 0)
        {
            const size_t equals = word.find('=');
            if (equals != std::string::npos)
            {
                args[word.substr(2, equals - 2)] = word.substr(equals + 1);
                pendingKey.clear();
            }
            else
            {
                pendingKey = word.substr(2);
                args[pendingKey] = "1";   // a flag, unless a value follows
            }
        }
        else if (!pendingKey.empty())
        {
            args[pendingKey] = word;
            pendingKey.clear();
        }
    }
    return args;
}

std::string jsonString(const std::string& text)
{
    std::string out = "\"";
    for (char c : text)
    {
        if (c == '"' || c == '\\') out += '\\', out += c;
        else if (c == '\n') out += "\\n";
        else if ((unsigned char)c < 0x20) out += ' ';
        else out += c;
    }
    return out + "\"";
}

// ------------------------------------------------------------------------------------------------
// Presentation: a swapchain on the XComponent's window at the render resolution (the compositor
// scales it to the component), filled by copying the Display pass's 8-bit pixels into its images.
class Presenter
{
public:
    Presenter(Device& dev, OHNativeWindow* window, uint32_t width, uint32_t height) : mDev(dev), mWidth(width), mHeight(height)
    {
        // Render at the bake's resolution; the compositor scales to the component.
        const int32_t geometry = OH_NativeWindow_NativeWindowHandleOpt(window, SET_BUFFER_GEOMETRY, (int32_t)width, (int32_t)height);
        VkSurfaceCreateInfoOHOS sci = {VK_STRUCTURE_TYPE_SURFACE_CREATE_INFO_OHOS};
        sci.window = window;
        VK_CHECK(vkCreateSurfaceOHOS(dev.instance, &sci, nullptr, &mSurface));
        VkBool32 canPresent = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(dev.physical, dev.queueFamily, mSurface, &canPresent);
        if (!canPresent) fail("the compute queue cannot present to the XComponent's surface");

        VkSurfaceCapabilitiesKHR caps;
        VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev.physical, mSurface, &caps));
        uint32_t count = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(dev.physical, mSurface, &count, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(count);
        vkGetPhysicalDeviceSurfaceFormatsKHR(dev.physical, mSurface, &count, formats.data());
        // Four 8-bit channels, which the pixels are copied into as they are: the Display pass writes
        // sRGB-encoded values, right for a UNORM and for an SRGB swapchain alike.
        const VkFormat wanted[] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_B8G8R8A8_SRGB};
        VkSurfaceFormatKHR chosen = {VK_FORMAT_UNDEFINED, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
        for (VkFormat want : wanted)
        {
            for (const VkSurfaceFormatKHR& f : formats)
                if (f.format == want && chosen.format == VK_FORMAT_UNDEFINED) chosen = f;
        }
        if (chosen.format == VK_FORMAT_UNDEFINED) fail("the surface offers no 8-bit RGBA or BGRA format");
        mBgra = chosen.format == VK_FORMAT_B8G8R8A8_UNORM || chosen.format == VK_FORMAT_B8G8R8A8_SRGB;
        if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) fail("swapchain images cannot be copied into");
        mExtent = caps.currentExtent.width == 0xFFFFFFFFu ? VkExtent2D{width, height} : caps.currentExtent;
        std::printf("SET_BUFFER_GEOMETRY %u x %u -> %d; surface: extent %ux%u, transform 0x%x (supported 0x%x), format %d, %u images\n", width, height,
                    geometry, mExtent.width, mExtent.height, (unsigned)caps.currentTransform, (unsigned)caps.supportedTransforms, (int)chosen.format,
                    caps.minImageCount);
        if (mExtent.width != width || mExtent.height != height)
            std::printf("note: the swapchain is %ux%u, the image %ux%u: only the overlap is shown\n", mExtent.width, mExtent.height, width, height);

        VkSwapchainCreateInfoKHR ci = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        ci.surface = mSurface;
        ci.minImageCount = std::max(caps.minImageCount, 2u);
        if (caps.maxImageCount > 0) ci.minImageCount = std::min(ci.minImageCount, caps.maxImageCount);
        ci.imageFormat = chosen.format;
        ci.imageColorSpace = chosen.colorSpace;
        ci.imageExtent = mExtent;
        ci.imageArrayLayers = 1;
        ci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        // The image is not pre-rotated: the compositor applies the display's orientation.
        ci.preTransform = (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR : caps.currentTransform;
        const VkCompositeAlphaFlagBitsKHR alphas[] = {VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
                                                      VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR, VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR};
        ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        for (VkCompositeAlphaFlagBitsKHR a : alphas)
        {
            if (caps.supportedCompositeAlpha & a)
            {
                ci.compositeAlpha = a;
                break;
            }
        }
        ci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        ci.clipped = VK_TRUE;
        VK_CHECK(vkCreateSwapchainKHR(dev.device, &ci, nullptr, &mSwapchain));
        vkGetSwapchainImagesKHR(dev.device, mSwapchain, &count, nullptr);
        mImages.resize(count);
        vkGetSwapchainImagesKHR(dev.device, mSwapchain, &count, mImages.data());

        VkSemaphoreCreateInfo si = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VK_CHECK(vkCreateSemaphore(dev.device, &si, nullptr, &mAcquired));
        // One per image: a present's wait on its semaphore may still be pending when the next frame
        // signals; the image is only handed out again once that present is done.
        mPresentable.resize(count);
        for (VkSemaphore& s : mPresentable) VK_CHECK(vkCreateSemaphore(dev.device, &si, nullptr, &s));
        VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VK_CHECK(vkCreateFence(dev.device, &fi, nullptr, &mFence));
    }

    ~Presenter()
    {
        vkDeviceWaitIdle(mDev.device);
        vkDestroyFence(mDev.device, mFence, nullptr);
        for (VkSemaphore s : mPresentable) vkDestroySemaphore(mDev.device, s, nullptr);
        vkDestroySemaphore(mDev.device, mAcquired, nullptr);
        vkDestroySwapchainKHR(mDev.device, mSwapchain, nullptr);
        vkDestroySurfaceKHR(mDev.instance, mSurface, nullptr);
    }

    bool bgra() const { return mBgra; }
    uint32_t width() const { return mWidth; }
    uint32_t height() const { return mHeight; }

    /// Shows the frame. False when the swapchain has to be made anew.
    bool present(const Display& display)
    {
        uint32_t index = 0;
        VkResult result = vkAcquireNextImageKHR(mDev.device, mSwapchain, UINT64_MAX, mAcquired, VK_NULL_HANDLE, &index);
        if (result == VK_ERROR_OUT_OF_DATE_KHR) return false;
        if (result != VK_SUBOPTIMAL_KHR) VK_CHECK(result);

        VkCommandBuffer cb = mDev.begin();
        display.record(cb);
        VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = mImages[index];
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;   // the whole image is overwritten
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        VkBufferImageCopy region = {};
        region.bufferRowLength = display.width();
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {std::min(mExtent.width, display.width()), std::min(mExtent.height, display.height()), 1};
        vkCmdCopyBufferToImage(cb, display.pixels().buffer, mImages[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = 0;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        VK_CHECK(vkEndCommandBuffer(cb));

        const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &mAcquired;
        submit.pWaitDstStageMask = &waitStage;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cb;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &mPresentable[index];
        VK_CHECK(vkQueueSubmit(mDev.queue, 1, &submit, mFence));
        VK_CHECK(vkWaitForFences(mDev.device, 1, &mFence, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(mDev.device, 1, &mFence));
        vkFreeCommandBuffers(mDev.device, mDev.pool, 1, &cb);

        VkPresentInfoKHR info = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        info.waitSemaphoreCount = 1;
        info.pWaitSemaphores = &mPresentable[index];
        info.swapchainCount = 1;
        info.pSwapchains = &mSwapchain;
        info.pImageIndices = &index;
        result = vkQueuePresentKHR(mDev.queue, &info);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) return false;
        VK_CHECK(result);
        return true;
    }

private:
    Device& mDev;
    uint32_t mWidth, mHeight;
    VkSurfaceKHR mSurface = VK_NULL_HANDLE;
    VkSwapchainKHR mSwapchain = VK_NULL_HANDLE;
    VkExtent2D mExtent = {};
    bool mBgra = false;
    std::vector<VkImage> mImages;
    VkSemaphore mAcquired = VK_NULL_HANDLE;
    std::vector<VkSemaphore> mPresentable;
    VkFence mFence = VK_NULL_HANDLE;
};

// ------------------------------------------------------------------------------------------------
// The camera under two thumbs, as Falcor's first-person controller moves it with a gamepad
// (FirstPersonCameraControllerCommon::update): the left stick moves along the view direction (y) and
// sideways (x) and the triggers up and down, `distance` (speed times time) for a full stick; the right
// thumb turns the view, yaw right and pitch down for positive angles, as the mouse does on the desktop.
// The stick gets the gamepad's dead zone and power curve.
struct CameraMove
{
    double stickX = 0.0, stickY = 0.0, lift = 0.0, yaw = 0.0, pitch = 0.0;
    bool any() const { return stickX != 0.0 || stickY != 0.0 || lift != 0.0 || yaw != 0.0 || pitch != 0.0; }
};

void flyCamera(CameraPose& pose, const CameraMove& move, double distance)
{
    auto dot = [](const double* a, const double* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
    auto cross = [](const double* a, const double* b, double* out) {
        out[0] = a[1] * b[2] - a[2] * b[1];
        out[1] = a[2] * b[0] - a[0] * b[2];
        out[2] = a[0] * b[1] - a[1] * b[0];
    };
    auto normalize = [&](double* v) {
        const double length = std::sqrt(dot(v, v));
        if (length > 0.0)
            for (int i = 0; i < 3; i++) v[i] /= length;
    };
    // Rodrigues: v turned by `angle` about the unit axis.
    auto rotate = [&](double* v, const double* axis, double angle) {
        double c[3];
        cross(axis, v, c);
        const double d = dot(axis, v), cosA = std::cos(angle), sinA = std::sin(angle);
        for (int i = 0; i < 3; i++) v[i] = v[i] * cosA + c[i] * sinA + axis[i] * d * (1.0 - cosA);
    };
    double position[3], forward[3], up[3], right[3];
    for (int i = 0; i < 3; i++)
    {
        position[i] = pose.position[i];
        forward[i] = (double)pose.target[i] - pose.position[i];
        up[i] = pose.up[i];
    }
    const double targetDistance = std::max(std::sqrt(dot(forward, forward)), 1e-6);
    normalize(forward);
    normalize(up);
    // Yaw about the up vector: a positive angle turns the view to the right.
    rotate(forward, up, -move.yaw);
    cross(forward, up, right);
    normalize(right);
    // Pitch about the right vector, down for a positive angle; looking straight up or down is where
    // the view would flip: stop 8 degrees short of it.
    const double elevation = std::asin(std::clamp(dot(forward, up), -1.0, 1.0)), limit = 1.43;
    const double pitch = std::clamp(elevation - move.pitch, -limit, limit) - elevation;
    rotate(forward, right, pitch);
    normalize(forward);
    cross(forward, up, right);
    normalize(right);

    // The stick as Falcor treats a gamepad's: dead zone 0.1, then a power curve of 1.2 per axis.
    double sx = move.stickX, sy = move.stickY;
    const double length = std::sqrt(sx * sx + sy * sy), kDeadZone = 0.1, kPower = 1.2;
    const double scale = length > 0.0 ? std::max(length - kDeadZone, 0.0) / (1.0 - kDeadZone) : 0.0;
    sx *= scale;
    sy *= scale;
    sx = std::pow(std::fabs(sx), kPower) * (sx < 0.0 ? -1.0 : 1.0);
    sy = std::pow(std::fabs(sy), kPower) * (sy < 0.0 ? -1.0 : 1.0);
    for (int i = 0; i < 3; i++)
    {
        // Pushing the stick up (y negative, as the screen counts) moves forward.
        position[i] += (forward[i] * -sy + right[i] * sx + up[i] * move.lift) * distance;
        pose.position[i] = (float)position[i];
        pose.target[i] = (float)(position[i] + forward[i] * targetDistance);
    }
}

// ------------------------------------------------------------------------------------------------
// The render thread.
void saveResults(Runtime& runtime, const Display& display, bool bgra, const std::string& dir, bool everything)
{
    makeDirectories(dir);
    // display.bin in RGBA order, as the replayer writes it, whatever order the swapchain wants.
    std::vector<uint8_t> pixels(display.pixels().as<uint8_t>(), display.pixels().as<uint8_t>() + (size_t)display.width() * display.height() * 4);
    if (bgra)
        for (size_t i = 0; i < pixels.size(); i += 4) std::swap(pixels[i], pixels[i + 2]);
    writeFile(dir + "/display.bin", pixels.data(), pixels.size());
    if (runtime.trainSteps() > 0)
    {
        runtime.saveCheckpoint(dir + "/trained.ckpt");
        const TrainSchedule& schedule = runtime.schedule();
        std::string text = format("frames %u\nheat %.9g\nholdLeft %u\nlearningRate %.9g\nemaDecay %.9g\n", runtime.framesDone(), schedule.heat,
                                  schedule.holdLeft, schedule.learningRate, schedule.emaDecay);
        if (runtime.clusterReuse())
        {
            // The cluster passes' CPU-side counters, which train_compare.py checks against Falcor's.
            const replay::ClusterClock& k = runtime.clusterClock();
            text += format("visibilityFrame %u\ngeometryMovedPrevious %u\nfrequencyFrames %u\nmotionFarWeightFrames %u\n", k.visibilityFrame,
                           k.geometryMovedPrevious ? 1u : 0u, k.frequencyFrames, k.motionFarWeightFrames);
        }
        writeFile(dir + "/schedule.txt", text.data(), text.size());
    }
    if (everything) runtime.writeResults(dir);
    std::printf("saved the results of frame %u to %s\n", runtime.framesDone(), dir.c_str());
}

/// What the render thread keeps from one scene to the next.
struct App
{
    std::string filesDir, assets;
    std::map<std::string, std::string> args;
    RuntimeOptions options;
    uint32_t logEvery = 30, saveAt = 0;
    bool saveEverything = false;
    double fixedStep = -1.0;   // --dt: seconds of animation per frame; < 0: the clock
    Device dev;
    std::unique_ptr<Presenter> presenter;
    std::vector<SceneEntry> scenes;
    std::string wanted;        // scene/variant to run next
    std::string model = "bake";
    std::string message;       // the outcome of the last command, for the panel
    uint32_t moveSpeed = 1;    // index into kMoveSpeeds

    std::string arg(const char* name, const std::string& fallback) const
    {
        auto it = args.find(name);
        return it == args.end() ? fallback : it->second;
    }

    const SceneEntry* find(const std::string& key) const
    {
        for (const SceneEntry& e : scenes)
            if (e.key() == key) return &e;
        return nullptr;
    }

    std::string modelPath(const SceneEntry& scene) const { return filesDir + "/models/" + scene.scene + "-" + scene.variant + ".ckpt"; }
};

/// The part of the panel's state that does not depend on a running scene.
std::string baseState(const App& app, const std::string& status, const SceneEntry* current, bool loaded)
{
    std::string json = "{\"status\":" + jsonString(status) + ",\"message\":" + jsonString(app.message) + ",\"scenes\":[";
    std::string last;
    bool first = true;
    for (const SceneEntry& e : app.scenes)
    {
        if (e.scene != last)
        {
            json += std::string(first ? "" : "]},") + "{\"name\":" + jsonString(e.scene) + ",\"variants\":[" + jsonString(e.variant);
            first = false;
            last = e.scene;
        }
        else
        {
            json += "," + jsonString(e.variant);
        }
    }
    json += first ? "]" : "]}]";
    json += ",\"scene\":" + jsonString(current ? current->scene : "") + ",\"variant\":" + jsonString(current ? current->variant : "");
    json += std::string(",\"loaded\":") + (loaded ? "true" : "false");
    return json;
}

void publish(const std::string& json, const std::string& status)
{
    std::lock_guard<std::mutex> lock(gShared.mutex);
    gShared.state = json;
    gShared.status = status;
}

/// Takes what the panel asked for since the last frame, and the camera controls: the stick and the
/// lift as they are now, the look turned since the last call.
void takeInput(std::vector<std::pair<std::string, double>>& sets, std::vector<std::pair<std::string, std::string>>& commands, CameraMove& move)
{
    std::lock_guard<std::mutex> lock(gShared.mutex);
    sets.swap(gShared.sets);
    commands.swap(gShared.commands);
    move = {gShared.stickX, gShared.stickY, gShared.lift, gShared.lookYaw, gShared.lookPitch};
    gShared.lookYaw = gShared.lookPitch = 0.0;
}

/// The speed factors of the panel's "Move Speed" (Falcor: Ctrl 0.25, Shift 10).
constexpr double kMoveSpeeds[] = {0.25, 1.0, 3.0, 10.0};

/// Commands that do not need a running scene. True if it was one.
bool appCommand(App& app, const std::string& name, const std::string& argument)
{
    if (name == "scene")
    {
        app.wanted = argument;
        app.model = "bake";
        return true;
    }
    if (name == "rescan")
    {
        app.scenes = findScenes(app.filesDir);
        return true;
    }
    return false;
}

/// Runs one scene until the panel asks for another one (or for this one again). Throws on errors.
void runScene(App& app, const SceneEntry& scene)
{
    const std::string name = scene.scene + " (" + scene.variant + ")";
    publish(baseState(app, "loading " + name, &scene, false) + "}", "loading " + name + ": scene data, acceleration structures, pipelines, NPU models");
    const auto loadStart = std::chrono::steady_clock::now();
    Runtime runtime(app.dev, scene.dir, app.model, app.options);
    std::printf("scene %s loaded in %.1f s (model: %s)\n", name.c_str(), msSince(loadStart) / 1000.0, app.model.c_str());
    // Settings given at launch (--set=name:value,...), for scripted runs.
    {
        std::istringstream pairs(app.arg("set", ""));
        std::string pair;
        while (std::getline(pairs, pair, ','))
        {
            const size_t colon = pair.find(':');
            if (colon == std::string::npos || !runtime.set(pair.substr(0, colon), std::stod(pair.substr(colon + 1)))) fail("--set: no setting " + pair);
        }
    }
    writeFile(app.filesDir + "/last_scene.txt", scene.key().data(), scene.key().size());
    const std::string title = format("%s %ux%u, %s", name.c_str(), runtime.width(), runtime.height(),
                                     app.options.networks == "npu" ? "MLPs on the NPU" : "MLPs on the CPU");
    const CameraPose bakedCamera = runtime.camera();
    // The scene's camera speed on the desktop (Scene::getCameraSpeed; bakes before it was recorded
    // have none: Falcor's default).
    const double cameraSpeed = runtime.bakeValue("cameraSpeed", 1.0);

    std::unique_ptr<Display> display;
    bool displayBgra = false;   // the channel order the Display pass writes, the swapchain's
    double msSum = 0.0, presentSum = 0.0, shownMs = 0.0;
    uint32_t msCount = 0;
    auto lastFrame = std::chrono::steady_clock::now();
    bool firstFrame = true;

    for (;;)
    {
        // The window: let go of it when the UI thread asks, take it when there is one.
        OHNativeWindow* window = nullptr;
        bool foreground = true, release = false;
        {
            std::lock_guard<std::mutex> lock(gShared.mutex);
            release = gShared.surfaceGone && gShared.windowClaimed;
            window = gShared.surfaceGone ? nullptr : gShared.window;
            foreground = gShared.foreground;
            if (!app.presenter && window && foreground) gShared.windowClaimed = true;   // claimed before it is used
        }
        if (release)
        {
            app.presenter.reset();
            std::lock_guard<std::mutex> lock(gShared.mutex);
            gShared.windowClaimed = false;
            gShared.changed.notify_all();
            std::printf("surface released\n");
            continue;
        }
        if (!window || !foreground)
        {
            // In the background, or no surface: neither the GPU nor the NPU is used.
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            lastFrame = std::chrono::steady_clock::now();
            continue;
        }
        if (app.presenter && (app.presenter->width() != runtime.width() || app.presenter->height() != runtime.height())) app.presenter.reset();
        if (!app.presenter) app.presenter = std::make_unique<Presenter>(app.dev, window, runtime.width(), runtime.height());
        if (!display || app.presenter->bgra() != displayBgra)
        {
            display.reset();
            display = std::make_unique<Display>(app.dev, app.options.shaderDir, runtime.outputImage(), app.presenter->bgra());
            displayBgra = app.presenter->bgra();
        }

        // What the panel asked for since the last frame.
        std::vector<std::pair<std::string, double>> sets;
        std::vector<std::pair<std::string, std::string>> commands;
        CameraMove move;
        takeInput(sets, commands, move);
        for (const auto& [setting, value] : sets)
        {
            if (setting == "moveSpeed")
                app.moveSpeed = std::min((uint32_t)std::max(value, 0.0), (uint32_t)(sizeof(kMoveSpeeds) / sizeof(kMoveSpeeds[0]) - 1));
            else if (!runtime.set(setting, value))
                std::printf("no setting named %s\n", setting.c_str());
        }
        bool leave = false;
        for (const auto& [command, argument] : commands)
        {
            if (appCommand(app, command, argument))
            {
                leave = leave || command == "scene";
            }
            else if (command == "resetSchedule")
            {
                runtime.schedule().reset();
                app.message = "training schedule reset";
            }
            else if (command == "clearHistory")
            {
                runtime.clearHistory();
                app.message = "cluster history cleared";
            }
            else if (command == "resetCamera")
            {
                runtime.setCamera(bakedCamera);
            }
            else if (command == "saveModel")
            {
                if (runtime.trainSteps() == 0)
                {
                    app.message = "nothing trained yet: no model saved";
                }
                else
                {
                    makeDirectories(app.filesDir + "/models");
                    runtime.saveCheckpoint(app.modelPath(scene));
                    app.message = "model saved: models/" + scene.scene + "-" + scene.variant + ".ckpt";
                }
            }
            else if (command == "loadModel")
            {
                if (fileExists(app.modelPath(scene)))
                {
                    app.wanted = scene.key();
                    app.model = app.modelPath(scene);
                    leave = true;
                }
                else
                {
                    app.message = "no saved model for this scene";
                }
            }
            else if (command == "resetModel")
            {
                app.wanted = scene.key();
                app.model = "bake";
                leave = true;
            }
            else
            {
                std::printf("unknown command %s\n", command.c_str());
            }
        }
        if (leave) return;

        // The animation follows the clock; a long gap (loading, the background) is not played back.
        // Falcor clamps its camera's step to 0.1 s, which at the phone's frame times would slow the
        // camera down: 0.5 s here.
        const auto now = std::chrono::steady_clock::now();
        double seconds = firstFrame ? 0.0 : std::min(std::chrono::duration<double>(now - lastFrame).count(), 0.5);
        if (app.fixedStep >= 0.0 && !firstFrame) seconds = app.fixedStep;
        lastFrame = now;
        firstFrame = false;

        // The camera moves while it does not follow its path (the page turns the path off when the
        // picture is touched).
        const bool freeCamera = !runtime.animated() || !runtime.settings().cameraAnimated;
        if (freeCamera && move.any())
        {
            CameraPose pose = runtime.camera();
            flyCamera(pose, move, cameraSpeed * kMoveSpeeds[app.moveSpeed] * seconds);
            runtime.setCamera(pose);
        }

        runtime.frame(seconds);
        const auto t0 = std::chrono::steady_clock::now();
        if (!app.presenter->present(*display)) app.presenter.reset();   // made anew in the next iteration
        const double presentMs = msSince(t0);
        if (runtime.settings().mode == RenderMode::Idle) std::this_thread::sleep_for(std::chrono::milliseconds(30));

        msSum += runtime.lastFrameMs();
        presentSum += presentMs;
        msCount++;
        shownMs = shownMs == 0.0 ? runtime.lastFrameMs() + presentMs : shownMs * 0.8 + (runtime.lastFrameMs() + presentMs) * 0.2;
        const uint32_t frame = runtime.framesDone();
        static const char* const kModes[] = {"idle", "render", "training", "training + render"};
        const RenderMode mode = runtime.settings().mode;
        const bool training = runtime.trains() && (mode == RenderMode::Train || mode == RenderMode::OnlineTrain);
        const std::string status = format("%s | frame %u | %.0f ms + %.0f ms present | %s", title.c_str(), frame, runtime.lastFrameMs(), presentMs,
                                          training || mode == RenderMode::Idle || mode == RenderMode::Render ? kModes[(uint32_t)mode] : "render (cannot train)");

        // The panel's state.
        const TrainSchedule& schedule = runtime.schedule();
        std::string json = baseState(app, status, &scene, true);
        json += format(",\"frame\":%u,\"frameMs\":%.2f,\"presentMs\":%.2f,\"smoothMs\":%.2f,\"trainSteps\":%u,\"events\":%u", runtime.bakedFrame() + frame,
                       runtime.lastFrameMs(), presentMs, shownMs, runtime.trainSteps(), runtime.lastEvents());
        json += format(",\"caps\":{\"animated\":%s,\"postFilter\":%s,\"lights\":%s,\"renderPath\":%s,\"train\":%s,\"savedModel\":%s,"
                       "\"clusterReuse\":%s}",
                       runtime.animated() ? "true" : "false", runtime.hasPostFilter() ? "true" : "false", runtime.hasLights() ? "true" : "false",
                       runtime.hasRenderPath() ? "true" : "false", runtime.trains() ? "true" : "false", fileExists(app.modelPath(scene)) ? "true" : "false",
                       runtime.clusterReuse() ? "true" : "false");
        json += ",\"settings\":{";
        bool firstSetting = true;
        for (const std::string& setting : Runtime::settingNames())
        {
            json += format("%s\"%s\":%.9g", firstSetting ? "" : ",", setting.c_str(), runtime.get(setting));
            firstSetting = false;
        }
        json += format(",\"moveSpeed\":%u", app.moveSpeed);
        json += format("},\"schedule\":{\"heat\":%.4f,\"holdLeft\":%u,\"learningRate\":%.6g,\"emaDecay\":%.6g}", schedule.heat, schedule.holdLeft,
                       schedule.learningRate, schedule.emaDecay);
        json += format(",\"anim\":{\"time\":%.3f,\"length\":%.3f,\"sample\":%u,\"samples\":%u}", runtime.animationTime(), runtime.animationLength(),
                       runtime.trackSample(), runtime.trackSamples());
        const CameraPose pose = runtime.camera();
        json += format(",\"camera\":[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f]", pose.position[0], pose.position[1], pose.position[2], pose.target[0], pose.target[1],
                       pose.target[2]);
        json += format(",\"info\":{\"width\":%u,\"height\":%u,\"batchSize\":%.0f,\"numRHS\":%.0f,\"specRays\":%.0f,\"networks\":%s,\"model\":%s,"
                       "\"cameraSpeed\":%.3f}",
                       runtime.width(), runtime.height(), runtime.bakeValue("lhs.size"), runtime.bakeValue("numRHS"), runtime.bakeValue("numSpecRays"),
                       jsonString(app.options.networks == "npu" ? "NPU" : "CPU").c_str(), jsonString(app.model == "bake" ? "fresh" : "saved").c_str(),
                       cameraSpeed * kMoveSpeeds[app.moveSpeed]);
        json += ",\"timings\":[";
        bool firstTiming = true;
        for (const auto& [what, ms] : runtime.lastTimings())
        {
            json += format("%s{\"name\":%s,\"ms\":%.2f}", firstTiming ? "" : ",", jsonString(what).c_str(), ms);
            firstTiming = false;
        }
        json += "]}";
        publish(json, status);

        if (frame == 1 || frame % app.logEvery == 0)
        {
            std::printf("frame %u: %.1f ms (%.1f ms on average over %u frames), present %.1f ms; track sample %u, events 0x%x, heat %.3f\n", frame,
                        runtime.lastFrameMs(), msSum / msCount, msCount, presentSum / msCount, runtime.trackSample(), runtime.lastEvents(), schedule.heat);
            for (const auto& [what, ms] : runtime.lastTimings()) std::printf("  %8.1f ms  %s\n", ms, what.c_str());
            msSum = presentSum = 0.0;
            msCount = 0;
        }
        if (frame == app.saveAt)
        {
            runtime.printSummary();
            saveResults(runtime, *display, displayBgra, app.filesDir + "/out", app.saveEverything);
        }
    }
}

void renderLoop(const std::string& filesDir, NativeResourceManager* resources, const std::string& launchArgs)
{
    std::printf("NCR app, launch arguments \"%s\"\n", launchArgs.c_str());
    App app;
    app.filesDir = filesDir;
    app.args = parseArgs(launchArgs);
    app.assets = filesDir + "/ncr";
    extractAssets(resources, app.assets);

    app.options.networks = app.arg("networks", "npu");
    app.options.shaderDir = app.assets + "/shaders";
    app.options.modelDir = app.assets + "/models";
    app.options.trainTile = (uint32_t)std::stoul(app.arg("train-tile", "4096"));
    app.options.noTrain = app.args.count("no-train") != 0;
    app.options.copyTiles = app.args.count("copy-tiles") != 0;
    app.options.cpuPrefixSum = app.args.count("cpu-prefix-sum") != 0;
    app.options.lossInterval = (uint32_t)std::stoul(app.arg("loss", "0"));
    app.logEvery = std::max(1u, (uint32_t)std::stoul(app.arg("log-every", "30")));
    app.saveAt = (uint32_t)std::stoul(app.arg("save-at", "0"));
    app.saveEverything = app.args.count("save-results") != 0;
    app.fixedStep = std::stod(app.arg("dt", "-1"));
    app.model = app.arg("model", "bake");

    setStatus("creating the Vulkan device");
    DeviceOptions deviceOptions;
    deviceOptions.instanceExtensions = {VK_KHR_SURFACE_EXTENSION_NAME, VK_OHOS_SURFACE_EXTENSION_NAME};
    deviceOptions.deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    app.dev.create(deviceOptions);
    std::printf("Vulkan device created: %s\n", app.dev.name);

    // The first scene: the one asked for, else the last run's, else the first there is.
    app.scenes = findScenes(filesDir);
    std::printf("%zu baked scenes on the phone\n", app.scenes.size());
    if (app.args.count("scene"))
    {
        const std::string scene = app.arg("scene", "");
        app.wanted = scene + "/" + app.arg("variant", "static");
        if (!app.find(app.wanted) && !app.args.count("variant") && app.find(scene + "/anim")) app.wanted = scene + "/anim";
    }
    else
    {
        app.wanted = readTextFile(filesDir + "/last_scene.txt");
        if (!app.find(app.wanted) && !app.scenes.empty()) app.wanted = app.scenes.front().key();
    }

    for (;;)
    {
        const SceneEntry* scene = app.find(app.wanted);
        std::string status;
        if (scene)
        {
            const SceneEntry current = *scene;
            try
            {
                runScene(app, current);   // returns when the panel asks for a scene
                vkDeviceWaitIdle(app.dev.device);
                continue;
            }
            catch (const std::exception& e)
            {
                std::printf("FAIL %s: %s\n", current.key().c_str(), e.what());
                status = "error in " + current.key() + ": " + e.what();
                app.model = "bake";
            }
        }
        else
        {
            status = app.scenes.empty() ? "no baked scene on the phone: bash ohos/app/tools/push_scenes.sh"
                                        : "no scene " + app.wanted + " on the phone: pick one in the panel";
        }
        // No scene runs: wait for the panel to pick one.
        app.wanted.clear();
        while (app.wanted.empty())
        {
            publish(baseState(app, status, nullptr, false) + "}", status);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            std::vector<std::pair<std::string, double>> sets;
            std::vector<std::pair<std::string, std::string>> commands;
            CameraMove move;
            takeInput(sets, commands, move);
            for (const auto& [command, argument] : commands) appCommand(app, command, argument);
            // The window may go away meanwhile.
            std::lock_guard<std::mutex> lock(gShared.mutex);
            if (gShared.surfaceGone && gShared.windowClaimed)
            {
                app.presenter.reset();
                gShared.windowClaimed = false;
                gShared.changed.notify_all();
            }
        }
    }
}

void* renderThread(void*)
{
    std::string filesDir, launchArgs;
    NativeResourceManager* resources = nullptr;
    {
        std::unique_lock<std::mutex> lock(gShared.mutex);
        if (!gShared.changed.wait_for(lock, std::chrono::seconds(10), [] { return gShared.initialised; }))
        {
            gShared.status = "error: init() was never called from ArkTS";
            return nullptr;
        }
        filesDir = gShared.filesDir;
        launchArgs = gShared.args;
        resources = gShared.resources;
    }
    // NPU calls from a thread with the default QoS are scheduled like background work.
    OH_QoS_SetThreadQoS(QOS_USER_INTERACTIVE);
    chdir(filesDir.c_str());
    try
    {
        renderLoop(filesDir, resources, launchArgs);
    }
    catch (const std::exception& e)
    {
        std::printf("FAIL %s\n", e.what());
        const std::string status = std::string("error: ") + e.what();
        publish("{\"status\":" + jsonString(status) + "}", status);
    }
    // No swapchain is left on the window: the UI thread need not wait for one.
    std::lock_guard<std::mutex> lock(gShared.mutex);
    gShared.windowClaimed = false;
    gShared.changed.notify_all();
    return nullptr;
}

void startRenderThread()
{
    pthread_attr_t attributes;
    pthread_attr_init(&attributes);
    pthread_attr_setstacksize(&attributes, 16u << 20);   // musl's default thread stack is small
    pthread_attr_setdetachstate(&attributes, PTHREAD_CREATE_DETACHED);
    pthread_t thread;
    pthread_create(&thread, &attributes, renderThread, nullptr);
    pthread_attr_destroy(&attributes);
}

// ------------------------------------------------------------------------------------------------
// XComponent callbacks (UI thread).
void onSurfaceCreated(OH_NativeXComponent* component, void* window)
{
    uint64_t width = 0, height = 0;
    OH_NativeXComponent_GetXComponentSize(component, window, &width, &height);
    std::lock_guard<std::mutex> lock(gShared.mutex);
    gShared.window = static_cast<OHNativeWindow*>(window);
    gShared.surfaceGone = false;
    std::printf("surface created: %llu x %llu\n", (unsigned long long)width, (unsigned long long)height);
    if (!gShared.threadStarted)
    {
        gShared.threadStarted = true;
        startRenderThread();
    }
    gShared.changed.notify_all();
}

void onSurfaceChanged(OH_NativeXComponent* component, void* window)
{
    uint64_t width = 0, height = 0;
    OH_NativeXComponent_GetXComponentSize(component, window, &width, &height);
    std::printf("surface changed: %llu x %llu\n", (unsigned long long)width, (unsigned long long)height);
}

void onSurfaceDestroyed(OH_NativeXComponent*, void*)
{
    // The swapchain must go before the window: wait until the render thread has let go of it.
    std::unique_lock<std::mutex> lock(gShared.mutex);
    gShared.surfaceGone = true;
    gShared.changed.notify_all();
    gShared.changed.wait_for(lock, std::chrono::seconds(5), [] { return !gShared.windowClaimed; });
    gShared.window = nullptr;
    gShared.surfaceGone = false;
    std::printf("surface destroyed\n");
}

// The page reads the touches (a layer over the picture, pages/Index.ets): the XComponent gets none.
void dispatchTouchEvent(OH_NativeXComponent*, void*) {}

// ------------------------------------------------------------------------------------------------
// NAPI.
std::string napiString(napi_env env, napi_value value)
{
    size_t length = 0;
    napi_get_value_string_utf8(env, value, nullptr, 0, &length);
    std::vector<char> text(length + 1, '\0');
    napi_get_value_string_utf8(env, value, text.data(), text.size(), &length);
    return std::string(text.data(), length);
}

napi_value napiInit(napi_env env, napi_callback_info info)
{
    size_t argc = 3;
    napi_value argv[3] = {nullptr, nullptr, nullptr};
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    if (argc < 3)
    {
        napi_throw_error(env, nullptr, "init(resourceManager, filesDir, args)");
        return nullptr;
    }
    NativeResourceManager* resources = OH_ResourceManager_InitNativeResourceManager(env, argv[0]);
    const std::string filesDir = napiString(env, argv[1]);
    const std::string launchArgs = napiString(env, argv[2]);
    startLogging(filesDir + "/ncr.log");
    std::lock_guard<std::mutex> lock(gShared.mutex);
    if (!gShared.initialised)
    {
        gShared.resources = resources;
        gShared.filesDir = filesDir;
        gShared.args = launchArgs;
        gShared.initialised = true;
        gShared.changed.notify_all();
    }
    return nullptr;
}

napi_value napiSetForeground(napi_env env, napi_callback_info info)
{
    size_t argc = 1;
    napi_value argv[1] = {nullptr};
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    bool foreground = true;
    if (argc >= 1) napi_get_value_bool(env, argv[0], &foreground);
    std::lock_guard<std::mutex> lock(gShared.mutex);
    gShared.foreground = foreground;
    gShared.changed.notify_all();
    return nullptr;
}

napi_value napiGetState(napi_env env, napi_callback_info)
{
    std::string state;
    {
        std::lock_guard<std::mutex> lock(gShared.mutex);
        state = gShared.state;
    }
    napi_value result = nullptr;
    napi_create_string_utf8(env, state.c_str(), state.size(), &result);
    return result;
}

napi_value napiSet(napi_env env, napi_callback_info info)
{
    size_t argc = 2;
    napi_value argv[2] = {nullptr, nullptr};
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    if (argc < 2) return nullptr;
    double value = 0.0;
    napi_get_value_double(env, argv[1], &value);
    std::lock_guard<std::mutex> lock(gShared.mutex);
    gShared.sets.push_back({napiString(env, argv[0]), value});
    return nullptr;
}

napi_value napiCommand(napi_env env, napi_callback_info info)
{
    size_t argc = 2;
    napi_value argv[2] = {nullptr, nullptr};
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    if (argc < 1) return nullptr;
    const std::string name = napiString(env, argv[0]);
    const std::string argument = argc >= 2 ? napiString(env, argv[1]) : std::string();
    std::lock_guard<std::mutex> lock(gShared.mutex);
    gShared.commands.push_back({name, argument});
    return nullptr;
}

/// Up to `count` numbers from a NAPI call; missing ones are 0.
void napiNumbers(napi_env env, napi_callback_info info, double* values, size_t count)
{
    size_t argc = 3;
    napi_value argv[3] = {nullptr, nullptr, nullptr};
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    for (size_t i = 0; i < count; i++)
    {
        values[i] = 0.0;
        if (i < argc) napi_get_value_double(env, argv[i], &values[i]);
    }
}

napi_value napiCameraStick(napi_env env, napi_callback_info info)
{
    double v[2];
    napiNumbers(env, info, v, 2);
    std::lock_guard<std::mutex> lock(gShared.mutex);
    gShared.stickX = std::clamp(v[0], -1.0, 1.0);
    gShared.stickY = std::clamp(v[1], -1.0, 1.0);
    return nullptr;
}

napi_value napiCameraLift(napi_env env, napi_callback_info info)
{
    double v[1];
    napiNumbers(env, info, v, 1);
    std::lock_guard<std::mutex> lock(gShared.mutex);
    gShared.lift = std::clamp(v[0], -1.0, 1.0);
    return nullptr;
}

napi_value napiCameraLook(napi_env env, napi_callback_info info)
{
    double v[2];
    napiNumbers(env, info, v, 2);
    std::lock_guard<std::mutex> lock(gShared.mutex);
    gShared.lookYaw += v[0];
    gShared.lookPitch += v[1];
    return nullptr;
}

napi_value moduleInit(napi_env env, napi_value exports)
{
    napi_property_descriptor properties[] = {
        {"init", nullptr, napiInit, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"setForeground", nullptr, napiSetForeground, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"getState", nullptr, napiGetState, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"set", nullptr, napiSet, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"command", nullptr, napiCommand, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"cameraStick", nullptr, napiCameraStick, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"cameraLift", nullptr, napiCameraLift, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"cameraLook", nullptr, napiCameraLook, nullptr, nullptr, nullptr, napi_default, nullptr},
    };
    napi_define_properties(env, exports, sizeof(properties) / sizeof(properties[0]), properties);
    // The library is loaded twice: by `import entry from 'libentry.so'` (no XComponent) and by the
    // XComponent (libraryname), which brings its native object.
    napi_value object = nullptr;
    if (napi_get_named_property(env, exports, OH_NATIVE_XCOMPONENT_OBJ, &object) == napi_ok && object)
    {
        OH_NativeXComponent* component = nullptr;
        if (napi_unwrap(env, object, reinterpret_cast<void**>(&component)) == napi_ok && component)
        {
            static OH_NativeXComponent_Callback callbacks = {onSurfaceCreated, onSurfaceChanged, onSurfaceDestroyed, dispatchTouchEvent};
            OH_NativeXComponent_RegisterCallback(component, &callbacks);
        }
    }
    return exports;
}

napi_module gModule = {1, 0, nullptr, moduleInit, "entry", nullptr, {nullptr, nullptr, nullptr, nullptr}};
}  // namespace

// In the shared library itself: a static library's constructor would be dropped by the linker.
extern "C" __attribute__((constructor)) void registerEntryModule()
{
    napi_module_register(&gModule);
}
