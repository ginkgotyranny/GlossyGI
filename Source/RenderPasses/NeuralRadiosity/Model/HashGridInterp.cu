#include "HashGridInterp.h"
#include <vector>
#include <cuda_runtime.h>
#include <assert.h>


__constant__ uint32_t cResolutions[MAX_LEVELS];
__constant__ uint32_t cGridSizes[MAX_LEVELS];
__constant__ uint32_t cGridOffsets[MAX_LEVELS];
__constant__ float cGridScales[MAX_LEVELS];


namespace HashGridInterp {

__device__ __forceinline__ uint3 _corner_offset(int c) {
    return make_uint3(c & 1, (c >> 1) & 1, (c >> 2) & 1);
}

__device__ __forceinline__ uint32_t _hash_index(uint3 index, int level, uint32_t log2HashMapSize) {
    const uint32_t maxHash = 1u << log2HashMapSize;

    uint32_t res1 = cResolutions[level] + 1;
    uint32_t denseMax = cGridSizes[level];

    uint32_t hashed = 0;

    if (denseMax > maxHash) {
        // Hash mode
        uint64_t result =
            index.x * PRIME_X +
            index.y * PRIME_Y +
            index.z * PRIME_Z;
        hashed = static_cast<uint32_t>(result & (maxHash - 1));
    } else {
        // Dense indexing mode
        // TODO: remove mod after applying bbox normalize
        hashed = ((res1 * index.x + index.y) * res1 + index.z) % denseMax;
    }

    return hashed;
}

template<int N_FEATURES_PER_LEVEL>
__global__ void forwardKernel(
    const precision_t* grids,
    precision_t* input,
    uint32_t count,
    uint32_t fullDim,
    uint32_t encOffset,
    uint32_t posOffset,
    uint32_t scaleOffset,
    uint32_t weightOffset,
    const Config config
) {
    uint32_t pixIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t clusterIdx = threadIdx.y;
    if (pixIdx >= count) return;

    precision_t* pixOut = input + pixIdx * fullDim + encOffset + clusterIdx * N_FEATURES_PER_LEVEL;
    const precision_t* posPtr = input + pixIdx * fullDim + posOffset + clusterIdx * 3;
    const precision_t* scalePtr = input + pixIdx * fullDim + scaleOffset + clusterIdx;
    const precision_t* weightPtr = input + pixIdx * fullDim + weightOffset + clusterIdx;

    float3 pos3 = make_float3((float)posPtr[0], (float)posPtr[1], (float)posPtr[2]);
    float scaleVal = (float)scalePtr[0];
    float clusterWeight = (float)weightPtr[0];

    // Get corresponding level and layer interpolation weight
    // Lower level means coarser, larger voxel size
    int upperLevel, lowerLevel;
    float upperLayerWeight = 0.0f;
    float lowerLayerWeight = 0.0f;

    {   // Upper(finer) level, fit bottom-up
        int level = config.nLevels;
        for (int i = 0; i < config.nLevels; ++i) {
            float voxelScale = cGridScales[i];
            if (scaleVal > voxelScale) {
                level = i;
                break;
            }
        }
        if (level == 0) {
            upperLayerWeight = 0.0f;
            upperLevel = 0;
        } else if (level == config.nLevels) {
            upperLayerWeight = 1.0f;
            upperLevel = config.nLevels - 1;
        } else {
            float coarserScale = cGridScales[level - 1];
            float finerScale = cGridScales[level];
            upperLayerWeight = (coarserScale - scaleVal) / (coarserScale - finerScale);
            upperLevel = level;
        }
    }
    {   // Lower(coarser) level, fit top-down
        int level = -1;
        for (int i = config.nLevels - 1; i >= 0; --i) {
            float voxelScale = cGridScales[i];
            if (scaleVal < voxelScale) {
                level = i;
                break;
            }
        }
        if (level == -1) {
            lowerLayerWeight = 1.0f;
            lowerLevel = 0;
        } else if (level == config.nLevels - 1) {
            lowerLayerWeight = 0.0f;
            lowerLevel = config.nLevels - 1;
        } else {
            float coarserScale = cGridScales[level];
            float finerScale = cGridScales[level + 1];
            lowerLayerWeight = (scaleVal - finerScale) / (coarserScale - finerScale);
            lowerLevel = level;
        }
    }

    const precision_t* upperGrid = grids + cGridOffsets[upperLevel] * N_FEATURES_PER_LEVEL;
    const precision_t* lowerGrid = grids + cGridOffsets[lowerLevel] * N_FEATURES_PER_LEVEL;
    
    uint32_t upperRes = cResolutions[upperLevel];
    uint32_t lowerRes = cResolutions[lowerLevel];
    
    float3 upperPosGrid = make_float3(pos3.x * upperRes, pos3.y * upperRes, pos3.z * upperRes);
    float3 lowerPosGrid = make_float3(pos3.x * lowerRes, pos3.y * lowerRes, pos3.z * lowerRes);
    uint3 upperBase = make_uint3(upperPosGrid.x, upperPosGrid.y, upperPosGrid.z);
    uint3 lowerBase = make_uint3(lowerPosGrid.x, lowerPosGrid.y, lowerPosGrid.z);
    float3 upperOffset = make_float3(upperPosGrid.x - upperBase.x, upperPosGrid.y - upperBase.y, upperPosGrid.z - upperBase.z);
    float3 lowerOffset = make_float3(lowerPosGrid.x - lowerBase.x, lowerPosGrid.y - lowerBase.y, lowerPosGrid.z - lowerBase.z);

    float upperEntry[MAX_FEATURES_PER_LEVEL];
    float lowerEntry[MAX_FEATURES_PER_LEVEL];
    #pragma unroll
    for (int i = 0; i < N_FEATURES_PER_LEVEL; i++) {
        upperEntry[i] = 0.0f;
        lowerEntry[i] = 0.0f;
    };

    #pragma unroll
    for (int corner = 0; corner < 8; corner++) {
        // Get hash index
        uint3 corner3 = _corner_offset(corner);
        uint3 upperIndex3 = make_uint3(upperBase.x + corner3.x, upperBase.y + corner3.y, upperBase.z + corner3.z);
        uint3 lowerIndex3 = make_uint3(lowerBase.x + corner3.x, lowerBase.y + corner3.y, lowerBase.z + corner3.z);
        uint32_t upperIndex = _hash_index(upperIndex3, upperLevel, config.log2HashMapSize);
        uint32_t lowerIndex = _hash_index(lowerIndex3, lowerLevel, config.log2HashMapSize);

        // Calculate interpolation weight
        float upperCornerWeight = (corner3.x ? upperOffset.x : (1 - upperOffset.x)) *
                            (corner3.y ? upperOffset.y : (1 - upperOffset.y)) *
                            (corner3.z ? upperOffset.z : (1 - upperOffset.z));
        float lowerCornerWeight = (corner3.x ? lowerOffset.x : (1 - lowerOffset.x)) *
                            (corner3.y ? lowerOffset.y : (1 - lowerOffset.y)) *
                            (corner3.z ? lowerOffset.z : (1 - lowerOffset.z));
        
        const precision_t* upperFeatures = upperGrid + upperIndex * N_FEATURES_PER_LEVEL;
        const precision_t* lowerFeatures = lowerGrid + lowerIndex * N_FEATURES_PER_LEVEL;

        #pragma unroll
        for (int i = 0; i < N_FEATURES_PER_LEVEL; i++) {
            upperEntry[i] += (float)upperFeatures[i] * upperCornerWeight;
            lowerEntry[i] += (float)lowerFeatures[i] * lowerCornerWeight;
        }
    }

    // Write to output
    #pragma unroll
    for (int i = 0; i < N_FEATURES_PER_LEVEL; i++) {
        pixOut[i] = (precision_t)((lowerEntry[i] * lowerLayerWeight + upperEntry[i] * upperLayerWeight) * clusterWeight);
    }
}

template<int N_FEATURES_PER_LEVEL>
__global__ void backwardKernel(
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
    const Config config
) {
    uint32_t pixIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t clusterIdx = threadIdx.y;
    if (pixIdx >= count) return;

    const precision_t* dL_dpixOut = dL_dinput + pixIdx * fullDim + encOffset + clusterIdx * N_FEATURES_PER_LEVEL;
    const precision_t* posPtr = input + pixIdx * fullDim + posOffset + clusterIdx * 3;
    const precision_t* scalePtr = input + pixIdx * fullDim + scaleOffset + clusterIdx;
    const precision_t* weightPtr = input + pixIdx * fullDim + weightOffset + clusterIdx;

    float3 pos3 = make_float3((float)posPtr[0], (float)posPtr[1], (float)posPtr[2]);
    float scaleVal = (float)scalePtr[0];
    float clusterWeight = (float)weightPtr[0];

    // Get corresponding level and layer interpolation weight
    // Lower level means coarser, larger voxel size
    int upperLevel, lowerLevel;
    float upperLayerWeight = 0.0f;
    float lowerLayerWeight = 0.0f;

    {   // Upper(finer) level, fit bottom-up
        int level = config.nLevels;
        for (int i = 0; i < config.nLevels; ++i) {
            float voxelScale = cGridScales[i];
            if (scaleVal > voxelScale) {
                level = i;
                break;
            }
        }
        if (level == 0) {
            upperLayerWeight = 0.0f;
            upperLevel = 0;
        } else if (level == config.nLevels) {
            upperLayerWeight = 1.0f;
            upperLevel = config.nLevels - 1;
        } else {
            float coarserScale = cGridScales[level - 1];
            float finerScale = cGridScales[level];
            upperLayerWeight = (coarserScale - scaleVal) / (coarserScale - finerScale);
            upperLevel = level;
        }
    }
    {   // Lower(coarser) level, fit top-down
        int level = -1;
        for (int i = config.nLevels - 1; i >= 0; --i) {
            float voxelScale = cGridScales[i];
            if (scaleVal < voxelScale) {
                level = i;
                break;
            }
        }
        if (level == -1) {
            lowerLayerWeight = 1.0f;
            lowerLevel = 0;
        } else if (level == config.nLevels - 1) {
            lowerLayerWeight = 0.0f;
            lowerLevel = config.nLevels - 1;
        } else {
            float coarserScale = cGridScales[level];
            float finerScale = cGridScales[level + 1];
            lowerLayerWeight = (scaleVal - finerScale) / (coarserScale - finerScale);
            lowerLevel = level;
        }
    }

    precision_t* dL_dupperGrid = dL_dgrids + cGridOffsets[upperLevel] * N_FEATURES_PER_LEVEL;
    precision_t* dL_dlowerGrid = dL_dgrids + cGridOffsets[lowerLevel] * N_FEATURES_PER_LEVEL;
    
    uint32_t upperRes = cResolutions[upperLevel];
    uint32_t lowerRes = cResolutions[lowerLevel];
    
    float3 upperPosGrid = make_float3(pos3.x * upperRes, pos3.y * upperRes, pos3.z * upperRes);
    float3 lowerPosGrid = make_float3(pos3.x * lowerRes, pos3.y * lowerRes, pos3.z * lowerRes);
    uint3 upperBase = make_uint3(upperPosGrid.x, upperPosGrid.y, upperPosGrid.z);
    uint3 lowerBase = make_uint3(lowerPosGrid.x, lowerPosGrid.y, lowerPosGrid.z);
    float3 upperOffset = make_float3(upperPosGrid.x - upperBase.x, upperPosGrid.y - upperBase.y, upperPosGrid.z - upperBase.z);
    float3 lowerOffset = make_float3(lowerPosGrid.x - lowerBase.x, lowerPosGrid.y - lowerBase.y, lowerPosGrid.z - lowerBase.z);

    // Write to output
    float dL_dupperEntry[MAX_FEATURES_PER_LEVEL];
    float dL_dlowerEntry[MAX_FEATURES_PER_LEVEL];
    #pragma unroll
    for (int i = 0; i < N_FEATURES_PER_LEVEL; i++) {
        dL_dupperEntry[i] = (float)dL_dpixOut[i] * upperLayerWeight * clusterWeight;
        dL_dlowerEntry[i] = (float)dL_dpixOut[i] * lowerLayerWeight * clusterWeight;
    }

    #pragma unroll
    for (int corner = 0; corner < 8; corner++) {
        // Get hash index
        uint3 corner3 = _corner_offset(corner);
        uint3 upperIndex3 = make_uint3(upperBase.x + corner3.x, upperBase.y + corner3.y, upperBase.z + corner3.z);
        uint3 lowerIndex3 = make_uint3(lowerBase.x + corner3.x, lowerBase.y + corner3.y, lowerBase.z + corner3.z);
        uint32_t upperIndex = _hash_index(upperIndex3, upperLevel, config.log2HashMapSize);
        uint32_t lowerIndex = _hash_index(lowerIndex3, lowerLevel, config.log2HashMapSize);

        // Calculate interpolation weight
        float upperCornerWeight = (corner3.x ? upperOffset.x : (1 - upperOffset.x)) *
                            (corner3.y ? upperOffset.y : (1 - upperOffset.y)) *
                            (corner3.z ? upperOffset.z : (1 - upperOffset.z));
        float lowerCornerWeight = (corner3.x ? lowerOffset.x : (1 - lowerOffset.x)) *
                            (corner3.y ? lowerOffset.y : (1 - lowerOffset.y)) *
                            (corner3.z ? lowerOffset.z : (1 - lowerOffset.z));
        
        precision_t* dL_dupperFeatures = dL_dupperGrid + upperIndex * N_FEATURES_PER_LEVEL;
        precision_t* dL_dlowerFeatures = dL_dlowerGrid + lowerIndex * N_FEATURES_PER_LEVEL;

        #pragma unroll
        for (int i = 0; i < N_FEATURES_PER_LEVEL; i++) {
            atomicAdd(&dL_dupperFeatures[i], (precision_t)(dL_dupperEntry[i] * upperCornerWeight));
            atomicAdd(&dL_dlowerFeatures[i], (precision_t)(dL_dlowerEntry[i] * lowerCornerWeight));
        }
    }
}

// Host functions
void initializeConstants(const Config& config) {
    float resolution = config.baseResolution;
    uint32_t offset = 0;
    std::vector<uint32_t> resolutions;
    std::vector<uint32_t> gridSizes;
    std::vector<uint32_t> gridOffsets;
    std::vector<float> gridScales;

    for (int i = 0; i < config.nLevels; i++) {
        uint64_t elems64 = (uint64_t) resolution + 1;
        elems64 = elems64 * elems64 * elems64;
        uint32_t elems = (uint32_t) std::min(elems64, (uint64_t) (1ull << config.log2HashMapSize));
        float scale = config.interpRatio / resolution;

        resolutions.push_back((uint32_t) resolution);
        gridSizes.push_back(elems);
        gridOffsets.push_back(offset);
        gridScales.push_back(scale);

        resolution *= config.perLevelScale;
        offset += elems;
    }

    cudaMemcpyToSymbol(cResolutions, resolutions.data(), config.nLevels * sizeof(uint32_t));
    cudaMemcpyToSymbol(cGridSizes, gridSizes.data(), config.nLevels * sizeof(uint32_t));
    cudaMemcpyToSymbol(cGridOffsets, gridOffsets.data(), config.nLevels * sizeof(uint32_t));
    cudaMemcpyToSymbol(cGridScales, gridScales.data(), config.nLevels * sizeof(float));

    assert(cudaGetLastError() == cudaSuccess && "Failed to set constants for HashGridInterp");
}

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
) {
    // Reserve threadIdx.y for cluster selection.
    const uint32_t blockSizeY = std::max(1u, config.nClusters);
    const uint32_t blockSizeX = std::max(1u, 256u / blockSizeY);
    dim3 block(blockSizeX, blockSizeY);
    dim3 grid((count + blockSizeX - 1) / blockSizeX);

    switch (config.nFeaturesPerLevel) {
        case 2: forwardKernel<2><<<grid, block, 0, stream>>>(grids, input, count, fullDim, encOffset, posOffset, scaleOffset, weightOffset, config); break;
        case 4: forwardKernel<4><<<grid, block, 0, stream>>>(grids, input, count, fullDim, encOffset, posOffset, scaleOffset, weightOffset, config); break;
        case 8: forwardKernel<8><<<grid, block, 0, stream>>>(grids, input, count, fullDim, encOffset, posOffset, scaleOffset, weightOffset, config); break;
        default: assert(false);
    }
}

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
) {
    // Reserve threadIdx.y for cluster selection.
    const uint32_t blockSizeY = std::max(1u, config.nClusters);
    const uint32_t blockSizeX = std::max(1u, 256u / blockSizeY);
    dim3 block(blockSizeX, blockSizeY);
    dim3 grid((count + blockSizeX - 1) / blockSizeX);

    switch (config.nFeaturesPerLevel) {
        case 2: backwardKernel<2><<<grid, block, 0, stream>>>(grids, input, dL_dinput, dL_dgrids, count, fullDim, encOffset, posOffset, scaleOffset, weightOffset, config); break;
        case 4: backwardKernel<4><<<grid, block, 0, stream>>>(grids, input, dL_dinput, dL_dgrids, count, fullDim, encOffset, posOffset, scaleOffset, weightOffset, config); break;
        case 8: backwardKernel<8><<<grid, block, 0, stream>>>(grids, input, dL_dinput, dL_dgrids, count, fullDim, encOffset, posOffset, scaleOffset, weightOffset, config); break;
        default: assert(false);
    }
}

}  // namespace HashGridInterp
