set(TARGET llama-kvstate-convert)
add_executable(${TARGET} kvstate-convert.cpp)
target_link_libraries(${TARGET} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
