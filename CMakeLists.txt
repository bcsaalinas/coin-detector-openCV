cmake_minimum_required(VERSION 3.20)

project(coin_detector
    VERSION 0.3.0
    DESCRIPTION "Coin detection and classification with OpenCV (milestones H0-H3)"
    LANGUAGES CXX)

# Default to Debug when no build type is given (single-config generators only).
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Only the modules needed through H3. H4 (SIFT) will add the features module:
# "features" in OpenCV 5.x, "features2d" in OpenCV 4.x.
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)
message(STATUS "OpenCV ${OpenCV_VERSION} found at ${OpenCV_DIR}")

add_executable(coin_detector
    src/main.cpp
    src/image_io.cpp
    src/regions.cpp
    src/preprocessing.cpp
    src/circle_detection.cpp
)

target_compile_features(coin_detector PRIVATE cxx_std_17)
set_target_properties(coin_detector PROPERTIES CXX_EXTENSIONS OFF)

target_include_directories(coin_detector PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(coin_detector PRIVATE ${OpenCV_LIBS})

# Default asset/output locations, resolved at configure time so the executable
# finds them no matter which directory it is launched from. Both can be
# overridden on the command line (--image, --output).
target_compile_definitions(coin_detector PRIVATE
    PROJECT_ROOT_DIR="${PROJECT_SOURCE_DIR}"
)

if(MSVC)
    target_compile_options(coin_detector PRIVATE /W4 /permissive-)
else()
    target_compile_options(coin_detector PRIVATE -Wall -Wextra -Wpedantic)
endif()
