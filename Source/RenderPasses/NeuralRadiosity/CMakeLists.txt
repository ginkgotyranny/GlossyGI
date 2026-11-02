add_plugin(NeuralRadiosity)
# Per-machine CUDA arch: 86 = Ampere (Linux box, default), 120 = Blackwell RTX 50 (Windows laptop).
# Override with -DNCR_CUDA_ARCH=<arch> at configure time; sm_86 SASS cannot load on Blackwell.
set(NCR_CUDA_ARCH 86 CACHE STRING "CUDA architecture for the NeuralRadiosity plugin")
set_property(TARGET NeuralRadiosity PROPERTY CUDA_ARCHITECTURES ${NCR_CUDA_ARCH})

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

# Slang includes NCRConfig.h, but the shader copy step only picks up slang/slangh.
set(NCR_CONFIG_SHADER_COPY ${FALCOR_SHADER_OUTPUT_DIRECTORY}/RenderPasses/NeuralRadiosity/NCRConfig.h)
add_custom_command(
    OUTPUT ${NCR_CONFIG_SHADER_COPY}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${FALCOR_SHADER_OUTPUT_DIRECTORY}/RenderPasses/NeuralRadiosity
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/NCRConfig.h ${NCR_CONFIG_SHADER_COPY}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/NCRConfig.h
    COMMENT "NeuralRadiosity: Copying NCRConfig.h"
)
add_custom_target(NeuralRadiosityConfigCopy DEPENDS ${NCR_CONFIG_SHADER_COPY})
add_dependencies(NeuralRadiosity NeuralRadiosityConfigCopy)

target_source_group(NeuralRadiosity "RenderPasses")
