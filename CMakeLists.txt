cmake_minimum_required(VERSION 3.20)

project(coin_detector
    VERSION 0.7.0
    DESCRIPTION "Coin detection and classification with OpenCV (still image + live camera)"
    LANGUAGES CXX)

# Default to Debug when no build type is given (single-config generators only).
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# SIFT lives in "features" (OpenCV 5) / "features2d" (OpenCV 4), and the RANSAC
# estimators in "geometry" (OpenCV 5) / "calib3d" (OpenCV 4). "videoio" reads
# the camera.
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio)
if(OpenCV_VERSION_MAJOR GREATER_EQUAL 5)
    find_package(OpenCV REQUIRED
        COMPONENTS core imgproc imgcodecs highgui videoio features geometry)
else()
    find_package(OpenCV REQUIRED
        COMPONENTS core imgproc imgcodecs highgui videoio features2d calib3d)
endif()
message(STATUS "OpenCV ${OpenCV_VERSION} found at ${OpenCV_DIR}")

# Everything except the two main() files, shared by both programs.
add_library(coin_core STATIC
    src/image_io.cpp
    src/regions.cpp
    src/preprocessing.cpp
    src/circle_detection.cpp
    src/size_filter.cpp
    src/color_filter.cpp
    src/coin_patch.cpp
    src/sift_features.cpp
    src/coin_matching.cpp
    src/report.cpp
    src/pipeline.cpp
    src/paper_view.cpp
    src/live_recognizer.cpp
)
target_compile_features(coin_core PUBLIC cxx_std_17)
target_include_directories(coin_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(coin_core PUBLIC ${OpenCV_LIBS})

# coin_detector: the still image, milestones H0-H6.
add_executable(coin_detector src/main.cpp)
# coin_camera: live recognition with a camera, milestone H7.
add_executable(coin_camera src/camera_main.cpp)

foreach(target coin_core coin_detector coin_camera)
    set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)
    # Default asset/output locations, resolved at configure time so the
    # programs find them from any working directory (overridable on the
    # command line).
    target_compile_definitions(${target} PRIVATE PROJECT_ROOT_DIR="${PROJECT_SOURCE_DIR}")
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endforeach()

target_link_libraries(coin_detector PRIVATE coin_core)
target_link_libraries(coin_camera PRIVATE coin_core)
