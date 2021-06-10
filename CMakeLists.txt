# Minimum version
cmake_minimum_required(VERSION 3.12.2)

# Project name
project(Project LANGUAGES C CUDA)
# Exec name
set(EXEC_NAME project)
set(CMAKE_C_FLAGS "-O3")

# CMake module
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# OpenMP
find_package(OpenMP REQUIRED)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
include_directories(${OPENMP_C_INCLUDE_DIRS})

# GLEW
find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})

# GLFW3
find_package(GLFW3 REQUIRED)
include_directories(${GLFW3_INCLUDE_DIR})

# OpenGL
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})

# Project files
include_directories(${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/inc)
file(GLOB C_SRCS
    "${PROJECT_SOURCE_DIR}/src/*.c"
)
file(GLOB CUDA_SRCS
    "${PROJECT_SOURCE_DIR}/src/*.cu"
)

# Compile options
option(EPSILON "epsilon to use." OFF)
option(ZETA "Bond number to use." OFF)
option(DELTA_T "Temporal Discretisation to use." OFF)
option(ETA "Stabilisation factor to use." OFF)
option(N_DISCR "Grid discretisation to use." OFF)

# Epsilon
if (NOT EPSILON)
    set(EPSILON 0.01f)
endif (NOT EPSILON)
message(STATUS "Using epsilon = ${EPSILON} .")
add_definitions(-DEPSILON=${EPSILON})

# zeta
if (NOT ZETA)
    set(ZETA 5.0f)
endif (NOT ZETA)
message(STATUS "Using Bond number = ${ZETA} .")
add_definitions(-DZETA=${ZETA})

# DELTA_T
if (NOT DELTA_T)
    set(DELTA_T 0.0005f)
endif (NOT DELTA_T)
message(STATUS "Using dt = ${DELTA_T} .")
add_definitions(-DDELTA_T=${DELTA_T})

# zeta
if (NOT ETA)
    set(ETA 0.0f)
endif (NOT ETA)
message(STATUS "Using stab coeff = ${ETA} .")
add_definitions(-DETA=${ETA})

# N_DISCR
if (NOT N_DISCR)
    set(N_DISCR 512)
endif (NOT N_DISCR)
message(STATUS "Using N_DISCR = ${N_DISCR} .")
add_definitions(-DN_DISCR=${N_DISCR})



# Use Gcc for compiling/linking the source files
add_executable(${EXEC_NAME} ${C_SRCS} ${CUDA_SRCS})
target_link_libraries(${EXEC_NAME} ${PYTHON_LIBRARIES} ${OPENMP_C_LIBRARIES} ${GLEW_LIBRARIES} ${GLFW3_LIBRARY} ${OPENGL_LIBRARY} pthread m)
set_property(TARGET ${EXEC_NAME} PROPERTY CUDA_ARCHITECTURES 70-real 72-virtual)
