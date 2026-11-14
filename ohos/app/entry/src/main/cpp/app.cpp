// NCR on HarmonyOS: the native side of the app.
//
//   ArkTS (EntryAbility, pages/Index)  ->  init(resourceManager, filesDir, args), setForeground, getStatus
//   XComponent                         ->  surface created / changed / destroyed (UI thread)
//   render thread                      ->  assets, Vulkan device, the NCR runtime (ohos/replay), and per
//                                          frame: runtime.frame(), the Display pass, a copy into the
//                                          swapchain image, present
//
// The runtime is the one the command-line replayer uses: the frames of a Falcor bake, rendered and
// trained on the phone, the MLPs on the NPU. Everything the native side prints (stdout, stderr) goes
// to <files>/ncr.log and to hilog (tag NCR); getStatus() gives the page one line of progress or the error.
//
// Launch arguments (one string, --name=value or --flag): --scene=<name> (default: the first scene
// in the assets), --networks=npu|tiles, --model=bake|<checkpoint path>, --train-tile=<n>,
// --no-train, --copy-tiles, --cpu-prefix-sum, --loss=<every n frames>, --log-every=<n frames>,
// --save-at=<frame>: after that frame write display.bin, trained.ckpt and schedule.txt to <files>/out
// (and with --save-results every image and ray-batch buffer, as the replayer does).
#include "ncr_runtime.h"

#include <ace/xcomponent/native_interface_xcomponent.h>
#include <hilog/log.h>
#include <napi/native_api.h>
#include <qos/qos.h>
#include <rawfile/raw_file.h>
#include <rawfile/raw_file_manager.h>

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

void makeDirectories(const std::string& path)
{
    for (size_t slash = path.find('/', 1); slash != std::string::npos; slash = path.find('/', slash + 1)) mkdir(path.substr(0, slash).c_str(), 0770);
    mkdir(path.c_str(), 0770);
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
    std::remove(installed.c_str());   // written last: an interrupted extraction is redone
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

/// The scenes in the assets, by the manifest: the folders under scenes/.
std::vector<std::string> sceneNames(const std::string& assets)
{
    std::vector<std::string> names;
    std::istringstream lines(readTextFile(assets + "/manifest.txt"));
    std::string line;
    while (std::getline(lines, line))
    {
        const size_t tab = line.find('\t');
        if (tab == std::string::npos || line.compare(tab + 1, 7, "scenes/") != 0) continue;
        const std::string rest = line.substr(tab + 8);
        const std::string name = rest.substr(0, rest.find('/'));
        if (std::find(names.begin(), names.end(), name) == names.end()) names.push_back(name);
    }
    std::sort(names.begin(), names.end());
    return names;
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

// ------------------------------------------------------------------------------------------------
// Presentation: a swapchain on the XComponent's window at the render resolution (the compositor
// scales it to the component), filled by copying the Display pass's 8-bit pixels into its images.
class Presenter
{
public:
    Presenter(Device& dev, OHNativeWindow* window, uint32_t width, uint32_t height) : mDev(dev)
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
        const std::string text = format("frames %u\nheat %.9g\nholdLeft %u\nlearningRate %.9g\nemaDecay %.9g\n", runtime.framesDone(), schedule.heat,
                                        schedule.holdLeft, schedule.learningRate, schedule.emaDecay);
        writeFile(dir + "/schedule.txt", text.data(), text.size());
    }
    if (everything) runtime.writeResults(dir);
    std::printf("saved the results of frame %u to %s\n", runtime.framesDone(), dir.c_str());
}

void renderLoop(const std::string& filesDir, NativeResourceManager* resources, const std::string& launchArgs)
{
    std::printf("NCR app, launch arguments \"%s\"\n", launchArgs.c_str());
    const std::map<std::string, std::string> args = parseArgs(launchArgs);
    auto arg = [&](const char* name, const std::string& fallback) {
        auto it = args.find(name);
        return it == args.end() ? fallback : it->second;
    };

    const std::string assets = filesDir + "/ncr";
    extractAssets(resources, assets);
    const std::vector<std::string> scenes = sceneNames(assets);
    if (scenes.empty()) fail("the assets hold no scene (ohos/app/tools/sync_assets.py)");
    const std::string scene = arg("scene", scenes.front());
    if (std::find(scenes.begin(), scenes.end(), scene) == scenes.end()) fail("no scene " + scene + " in the assets");

    RuntimeOptions options;
    options.networks = arg("networks", "npu");
    options.shaderDir = assets + "/shaders";
    options.modelDir = assets + "/models";
    options.trainTile = (uint32_t)std::stoul(arg("train-tile", "4096"));
    options.noTrain = args.count("no-train") != 0;
    options.copyTiles = args.count("copy-tiles") != 0;
    options.cpuPrefixSum = args.count("cpu-prefix-sum") != 0;
    options.lossInterval = (uint32_t)std::stoul(arg("loss", "0"));
    const uint32_t logEvery = std::max(1u, (uint32_t)std::stoul(arg("log-every", "30")));
    const uint32_t saveAt = (uint32_t)std::stoul(arg("save-at", "0"));
    const bool saveEverything = args.count("save-results") != 0;

    setStatus("creating the Vulkan device");
    Device dev;
    DeviceOptions deviceOptions;
    deviceOptions.instanceExtensions = {VK_KHR_SURFACE_EXTENSION_NAME, VK_OHOS_SURFACE_EXTENSION_NAME};
    deviceOptions.deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    dev.create(deviceOptions);
    std::printf("Vulkan device created: %s\n", dev.name);

    setStatus("loading " + scene + ": scene data, acceleration structures, pipelines, NPU models");
    Runtime runtime(dev, assets + "/scenes/" + scene, arg("model", "bake"), options);
    const std::string title = format("%s %ux%u, %s", scene.c_str(), runtime.width(), runtime.height(),
                                     options.networks == "npu" ? "MLPs on the NPU" : "MLPs on the CPU");
    std::unique_ptr<Presenter> presenter;
    std::unique_ptr<Display> display;
    bool displayBgra = false;   // the channel order the Display pass writes, the swapchain's
    double msSum = 0.0, presentSum = 0.0;
    uint32_t msCount = 0;

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
            if (!presenter && window && foreground) gShared.windowClaimed = true;   // claimed before it is used
        }
        if (release)
        {
            presenter.reset();
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
            continue;
        }
        if (!presenter)
        {
            presenter = std::make_unique<Presenter>(dev, window, runtime.width(), runtime.height());
            if (!display || presenter->bgra() != displayBgra)
            {
                display = std::make_unique<Display>(dev, options.shaderDir, runtime.outputImage(), presenter->bgra());
                displayBgra = presenter->bgra();
            }
        }

        runtime.frame();
        const auto t0 = std::chrono::steady_clock::now();
        if (!presenter->present(*display)) presenter.reset();   // made anew in the next iteration
        const double presentMs = msSince(t0);

        msSum += runtime.lastFrameMs();
        presentSum += presentMs;
        msCount++;
        const uint32_t frame = runtime.framesDone();
        setStatus(format("%s | frame %u | %.0f ms + %.0f ms present%s", title.c_str(), frame, runtime.lastFrameMs(), presentMs,
                         runtime.trains() ? " | training" : ""));
        if (frame == 1 || frame % logEvery == 0)
        {
            std::printf("frame %u: %.1f ms (%.1f ms on average over %u frames), present %.1f ms\n", frame, runtime.lastFrameMs(), msSum / msCount, msCount,
                        presentSum / msCount);
            for (const auto& [name, ms] : runtime.lastTimings()) std::printf("  %8.1f ms  %s\n", ms, name.c_str());
            msSum = presentSum = 0.0;
            msCount = 0;
        }
        if (frame == saveAt)
        {
            runtime.printSummary();
            saveResults(runtime, *display, displayBgra, filesDir + "/out", saveEverything);
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
        setStatus(std::string("error: ") + e.what());
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

napi_value napiGetStatus(napi_env env, napi_callback_info)
{
    std::string status;
    {
        std::lock_guard<std::mutex> lock(gShared.mutex);
        status = gShared.status;
    }
    napi_value result = nullptr;
    napi_create_string_utf8(env, status.c_str(), status.size(), &result);
    return result;
}

napi_value moduleInit(napi_env env, napi_value exports)
{
    napi_property_descriptor properties[] = {
        {"init", nullptr, napiInit, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"setForeground", nullptr, napiSetForeground, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"getStatus", nullptr, napiGetStatus, nullptr, nullptr, nullptr, napi_default, nullptr},
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
