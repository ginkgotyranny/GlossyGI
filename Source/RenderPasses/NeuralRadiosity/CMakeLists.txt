add_plugin(NeuralRadiosity)
# 120 = Blackwell RTX 50 (local Windows GPU); 86 = Ampere (Linux box).
set_property(TARGET NeuralRadiosity PROPERTY CUDA_ARCHITECTURES 120)

file(GLOB_RECURSE NR_SOURCES
    "*.cpp"
    "*.h"
    "*.cu"
    "*.slang"
)

target_sources(NeuralRadiosity PRIVATE ${NR_SOURCES})

target_link_libraries(NeuralRadiosity PRIVATE
    Falcor
    tiny-cuda-nn
    tiny-cuda-nn-resources
    fmt
)

target_copy_shaders(NeuralRadiosity RenderPasses/NeuralRadiosity)

target_source_group(NeuralRadiosity "RenderPasses")
