#pragma once
#include <cstdint>
#include <memory>
#include <cuda_runtime.h>
#include "ModelTypes.h"

#ifndef MAX_LEVELS
#define MAX_LEVELS 8
#endif

#ifndef MAX_FEATURES_PER_LEVEL
#define MAX_FEATURES_PER_LEVEL 16
#endif

#define PRIME_X 1
#define PRIME_Y 19349663
#define PRIME_Z 83492791


namespace HashGridInterp {
    struct Config {
        uint32_t nClusters;
        uint32_t nLevels;
        uint32_t nFeaturesPerLevel;
        uint32_t log2HashMapSize;
        uint32_t baseResolution;
        float perLevelScale;
        float interpRatio;

        // --- Debug: hash collision stats (same semantics as HashGrid::Config) ---
        uint32_t* counters = nullptr;
        uint32_t* slotMarker = nullptr;

        // --- Debug: level usage histogram ---
        // When != nullptr, forwardKernel atomically increments
        // histogram[upperLevel * nLevels + lowerLevel] for each (pixel, cluster)
        // query. Size must be nLevels * nLevels uint32_t. Reset per frame.
        uint32_t* levelHistogram = nullptr;
    };

    void initializeConstants(const Config& config);

    void launchForward(
        cudaStream_t stream,
        const precision_t* grids,
        precision_t* input,
        uint32_t count,
        uint32_t fullDim,
        uint32_t encOffset,
        uint32_t posOffset,
        uint32_t scaleOffset,
        uint32_t weightOffset,
        const Config& config
    );

    void launchBackward(
        cudaStream_t stream,
        const precision_t* grids,
        const precision_t* input,
        const precision_t* dL_dinput,
        precision_t* dL_dgrids,
        uint32_t count,
        uint32_t fullDim,
        uint32_t encOffset,
        uint32_t posOffset,
        uint32_t scaleOffset,
        uint32_t weightOffset,
        const Config& config
    );

    void resetCollisionStats(cudaStream_t stream, const Config& config);
    void resetLevelHistogram(cudaStream_t stream, const Config& config);
}
