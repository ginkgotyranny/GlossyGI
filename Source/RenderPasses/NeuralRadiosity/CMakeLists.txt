add_plugin(NeuralRadiosity)
set_property(TARGET NeuralRadiosity PROPERTY CUDA_ARCHITECTURES 86)

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
