cmake_minimum_required( VERSION 3.10.0 )
project( sph-benchmark )
enable_language( CUDA )
set( CMAKE_CXX_STANDARD 11 )
add_compile_options(-std=c++11)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# external glfw
set( GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE )
set( GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE )
set( GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE )
add_subdirectory( external/glfw )

# make sure we have OpenMP C++ support
find_package( OpenMP REQUIRED CXX )

# external includes
include_directories( SYSTEM "external/glm" )
include_directories( SYSTEM "external/glad/include" )
include_directories( SYSTEM "external/glfw/include" )


# application code
include_directories( "include" )

add_library(
    particles STATIC
  # "src/main.cpp"
    "src/particle.cu"
    "src/particle.h"
)
set_target_properties( particles
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON )

add_executable(
    particle_test 
      "src/main.cpp"
    #  "src/particle.h"
    "external/glad/src/glad.c" 
    )
set_property( TARGET particle_test 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON )

target_link_libraries(
    particle_test
    PRIVATE particles
    PRIVATE glfw
    PRIVATE OpenMP::OpenMP_CXX )
