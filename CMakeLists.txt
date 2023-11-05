cmake_minimum_required(VERSION 3.5.0)

project(ekf)
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/manif/include)
include_directories(${PROJECT_SOURCE_DIR}/manif/external/tl)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
message("EIGEN3_INCLUDE_DIR: ${EIGEN3_INCLUDE_DIR}")

add_executable(ekf_test test/ekf_test.cpp)

