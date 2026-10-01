set(TARGET llama-phone-kv-test)
add_executable(${TARGET} phone-kv-test.cpp)
target_link_libraries(${TARGET} PRIVATE llama ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

add_executable(fa-bias-test fa-bias-test.cpp)
target_link_libraries(fa-bias-test PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(fa-bias-test PRIVATE cxx_std_17)
