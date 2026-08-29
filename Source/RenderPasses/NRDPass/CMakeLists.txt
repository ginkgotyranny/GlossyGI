if(NOT (FALCOR_HAS_NRD AND (FALCOR_HAS_D3D12 OR FALCOR_HAS_VULKAN)))
    return()
endif()

add_plugin(NRDPass)

target_sources(NRDPass PRIVATE
    NRDPass.cpp
    NRDPass.h
    PackRadiance.cs.slang
)

target_link_libraries(NRDPass PRIVATE nrd)

if(FALCOR_HAS_VULKAN)
    find_package(Vulkan REQUIRED)
    target_link_libraries(NRDPass PRIVATE Vulkan::Vulkan)
endif()

if(FALCOR_LINUX)
    add_custom_command(TARGET NRDPass POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:nrd>
            ${FALCOR_OUTPUT_DIRECTORY}/libNRD.so.3.1.0
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            libNRD.so.3.1.0
            ${FALCOR_OUTPUT_DIRECTORY}/libNRD.so
    )
endif()

target_copy_shaders(NRDPass RenderPasses/NRDPass)

target_source_group(NRDPass "RenderPasses")
