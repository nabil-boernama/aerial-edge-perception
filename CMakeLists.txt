# Require a modern version of CMake
cmake_minimum_required(VERSION 3.15)

# Name the project and declare it as a C++ project
project(AerialEdgePerception LANGUAGES CXX)

# Force the compiler to use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Generate Position Independent Code (best practice for modern C++)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ==========================================
# 1. FIND OpenCV
# ==========================================
# This automatically finds OpenCV if you installed it via `sudo apt install libopencv-dev`
find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")

# ==========================================
# 2. CONFIGURE ONNX RUNTIME (Prebuilt Binaries)
# ==========================================
# Point CMake to the local onnxruntime folder in the project root
set(ORT_DIR "${CMAKE_SOURCE_DIR}/onnxruntime")

# Tell the compiler where the ONNX Runtime header files are
include_directories(${ORT_DIR}/include)

# Tell CMake exactly where to find the prebuilt shared library (.so)
find_library(ORT_LIB 
    NAMES onnxruntime 
    PATHS ${ORT_DIR}/lib 
    NO_DEFAULT_PATH
)

# Throw a loud error if the library is missing so we don't fail silently
if(NOT ORT_LIB)
    message(FATAL_ERROR "Could not find libonnxruntime.so in ${ORT_DIR}/lib. Ensure the folder structure is correct.")
endif()
message(STATUS "Found ONNX Runtime Library: ${ORT_LIB}")

# ==========================================
# 3. BUILD THE EXECUTABLE
# ==========================================
# Tell CMake which source files to compile into our final binary
add_executable(aerial-edge-perception 
    src/main.cpp 
    src/onnx_engine.cpp
)

# Tell the compiler where our local header files (inference_engine.hpp) are
target_include_directories(aerial-edge-perception PRIVATE include)

# Link OpenCV and ONNX Runtime to our executable
target_link_libraries(aerial-edge-perception 
    PRIVATE 
    ${OpenCV_LIBS}
    ${ORT_LIB}
)