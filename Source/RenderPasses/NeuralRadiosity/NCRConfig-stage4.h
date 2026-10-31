#ifndef NCR_CONFIG_H
#define NCR_CONFIG_H

// Shared by Slang and CUDA. Shader names (N_LEVELS, NUM_CLUSTERS, ...) stay
// overridable by getShaderDefines(); the numbers live only here.

// Sampling. Train/render ray counts are selected at runtime by the pass.
#define NCR_NUM_SPEC_RAYS_TRAIN 8
#define NCR_NUM_SPEC_RAYS_RENDER 32
#define NCR_NUM_KMEANS_ITERS_TRAIN 10
#define NCR_NUM_KMEANS_ITERS_RENDER 3
#define NCR_NUM_CLUSTERS 4
#define NCR_NUM_RHS_INIT 8
#define NCR_BATCH_SIZE_INIT (1u << 15)
#define NCR_RS_RR_PROB 0.4f
#define NCR_SPECULAR_ROUGHNESS_THRESHOLD 0.5f

// Primary hash grid.
#define NCR_N_LEVELS 6
#define NCR_N_FEATURES_PER_LEVEL 2
#define NCR_LOG2_HASHMAP_SIZE 18
#define NCR_BASE_RESOLUTION 32
#define NCR_PER_LEVEL_SCALE 2.0f

// Cluster hash grid.
#define NCR_N_LEVELS_SPEC 6
#define NCR_N_FEATURES_PER_LEVEL_SPEC 4
#define NCR_LOG2_HASHMAP_SIZE_SPEC 18
#define NCR_BASE_RESOLUTION_SPEC 16
#define NCR_PER_LEVEL_SCALE_SPEC 2.0f
#define NCR_INTERP_RATIO 1.0f

// Fused MLP. Output width stays 16 so it matches the fully fused layout.
#define NCR_DIFF_N_NEURONS 32
#define NCR_DIFF_N_HIDDEN_LAYERS 1
#define NCR_SPEC_N_NEURONS 32
#define NCR_SPEC_N_HIDDEN_LAYERS 2
#define NCR_OUTPUT_DIM 16
#define NCR_USE_FP16 1

// Optimizer.
#define NCR_ADAM_LR 5e-3f
#define NCR_EMA_DECAY 0.95f

#endif
