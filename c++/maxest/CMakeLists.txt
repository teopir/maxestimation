project(maxest)
cmake_minimum_required(VERSION 2.8.10)

#----------------------- CMAKE MODULES ------------------------

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

# ----------------------- INCLUDES ----------------------------

include_directories(include)

# ----------------------- LIBRARIES ---------------------------

find_package(BLAS REQUIRED)
find_package(Armadillo REQUIRED)
find_package(GSL REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIR}
    ${GSL_INCLUDE_DIRS}
    ${GSLCBLAS_INCLUDE_DIRS})

# ----------------------- GCC FLAGS ----------------------------

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")

# ------------------------ MESSAGES ----------------------------

message(STATUS "Build type       : " ${CMAKE_BUILD_TYPE})

# ------------------------ BUILD -------------------------------
file(GLOB_RECURSE _SOURCE src/*.cpp)
file(GLOB_RECURSE _INCLUDE include/*.h)

add_library(${PROJECT_NAME} ${_SOURCE} ${_INCLUDE})
target_link_libraries(${PROJECT_NAME}
    ${ARMADILLO_LIBRARIES}
    ${GSL_LIBRARY} ${BLAS_LIBRARIES} -lopenblas)

add_subdirectory(examples)
