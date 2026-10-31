#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <memory>
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

#define CONCAT  0
#define MEAN    1
#define INTERP  2

#ifndef LAYER_REDUCE
#define LAYER_REDUCE CONCAT
#endif


namespace HashGrid {
    struct Config {
        uint32_t nLevels;
        uint32_t nFeaturesPerLevel;
        uint32_t log2HashMapSize;
        uint32_t baseResolution;
        float perLevelScale;

        // --- Debug: hash collision stats ---
        // When != nullptr, forwardKernel records:
        //   counters[0] = number of unique hash slots touched this frame
        //   counters[1] = total hash-mode corner queries this frame
        // slotMarker must have (1 << log2HashMapSize) * nLevels uint32_t entries,
        // zeroed before each frame. Collisions = counters[1] - counters[0].
        uint32_t* counters = nullptr;
        uint32_t* slotMarker = nullptr;
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
        const Config& config
    );

    // Reset collision counters and slot marker buffer to zero.
    // gridSize = (1 << log2HashMapSize) * nLevels.
    void resetCollisionStats(cudaStream_t stream, const Config& config);
}
