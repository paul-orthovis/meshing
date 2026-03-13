cmake_minimum_required(VERSION 3.12)
project(cgal_tets)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CGAL REQUIRED COMPONENTS Core)
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

add_executable(remesh_obj remesh_obj.cpp)
target_link_libraries(remesh_obj CGAL::CGAL Eigen3::Eigen)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(remesh_obj PRIVATE -g -O0)
else()
    target_compile_options(remesh_obj PRIVATE -O3)
endif()
