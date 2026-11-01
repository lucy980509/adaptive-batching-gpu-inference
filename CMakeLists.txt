cmake_minimum_required(VERSION 3.20)

project(AdaptiveBatchingInference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/utf-8)
endif()

set(OpenCV_DIR
    "C:/opencv/opencv/build/x64/vc16/lib"
)

set(ONNXRUNTIME_DIR
    "C:/onnxruntime/onnxruntime-win-x64-gpu-1.26.0"
)

set(CUDA_BIN_DIR
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin"
)

set(CUDNN_BIN_DIR
    "C:/Program Files/NVIDIA/CUDNN/v9.8/bin/12.8"
)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

add_executable(p3_inference
    src/main.cpp
    src/preprocess.cpp
    src/inference_engine.cpp
    src/request_generator.cpp
    src/static_batcher.cpp
    src/adaptive_batcher.cpp
    src/slo_batcher.cpp
    src/metrics.cpp
)

target_include_directories(p3_inference PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${ONNXRUNTIME_DIR}/include"
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(p3_inference PRIVATE
    Threads::Threads
    ${OpenCV_LIBS}
    "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib"
)

file(GLOB RUNTIME_DLLS
    "${ONNXRUNTIME_DIR}/lib/*.dll"
    "C:/opencv/opencv/build/x64/vc16/bin/opencv_world4140.dll"
    "${CUDA_BIN_DIR}/*.dll"
    "${CUDNN_BIN_DIR}/*.dll"
)

foreach(DLL_FILE IN LISTS RUNTIME_DLLS)
    add_custom_command(
        TARGET p3_inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DLL_FILE}"
            "$<TARGET_FILE_DIR:p3_inference>"
        VERBATIM
    )
endforeach()

add_custom_command(
    TARGET p3_inference POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/models"
        "$<TARGET_FILE_DIR:p3_inference>/models"
    VERBATIM
)
