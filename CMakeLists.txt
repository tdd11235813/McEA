cmake_minimum_required(VERSION 3.0.2)

# Gives an error if you are doing in-source build.
# See https://cmake.org/Wiki/CMake_FAQ#Out-of-source_build_trees
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_DISABLE_SOURCE_CHANGES  ON)

# And optionally
set(CMAKE_COLOR_MAKEFILE ON)
#set(CMAKE_VERBOSE_MAKEFILE ON)

# set project build type
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE Release)
endif()
message(">> Setting up ${CMAKE_BUILD_TYPE} build")


add_subdirectory(./src/cpu/)
add_subdirectory(./src/gpu/)
