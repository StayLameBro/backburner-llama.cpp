set(TARGET llama-split-prefill)
add_executable(${TARGET} split-prefill.cpp)
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# correctness gate for the llama_state_filter API (split prefill of an append; see mirror-test.cpp)
add_executable(llama-mirror-test mirror-test.cpp)
target_link_libraries(llama-mirror-test PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(llama-mirror-test PRIVATE cxx_std_17)

# correctness gate for llama_set_ffn_offload (CPU reference FFN via Accelerate; see ffn-offload-test.cpp)
if (APPLE)
    # compiled as CXX with -x objective-c++: enable_language(OBJCXX) in a subdirectory makes CMake 4 classify
    # ggml-metal's .m files as OBJCXX with no compile rule (fresh configures then fail to link libggml-metal)
    add_executable(llama-ffn-offload-test ffn-offload-test.cpp ane-ffn.mm)
    set_source_files_properties(ane-ffn.mm PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-x objective-c++ -fobjc-arc")
    target_link_libraries(llama-ffn-offload-test PRIVATE llama ggml "-framework Accelerate" "-framework CoreML" "-framework Foundation" ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(llama-ffn-offload-test PRIVATE cxx_std_17)
endif()

# end-to-end gate for split prefill inside llama_decode (src/llama-split.cpp); see split-e2e-test.cpp
add_executable(llama-split-e2e-test split-e2e-test.cpp)
target_link_libraries(llama-split-e2e-test PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(llama-split-e2e-test PRIVATE cxx_std_17)
