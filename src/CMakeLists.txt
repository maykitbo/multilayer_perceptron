cmake_minimum_required(VERSION 3.5)

project(perceptron VERSION 1.0 LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER g++)

# find cblas
find_package(BLAS REQUIRED)

# gcc flags optimization
set(CMAKE_CXX_FLAGS "-O3 -march=native")

set(SOURCE_DIRS
    model
    common
    lib
)

foreach(dir ${SOURCE_DIRS})
    file(GLOB_RECURSE ONE_DIR_SOURCES 
        ${dir}/*.cc
        ${dir}/*.h
    )
    list(APPEND SOURCES ${ONE_DIR_SOURCES})
endforeach()

add_library(${PROJECT_NAME} STATIC ${SOURCES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${BLAS_LIBRARIES})

