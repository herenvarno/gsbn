cmake_minimum_required(VERSION 2.8.9)
project(gsbn)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)

# The project version number.
set(VERSION_MAJOR 1 CACHE STRING "Project major version number.")
set(VERSION_MINOR 5 CACHE STRING "Project minor version number.")
set(VERSION_PATCH 0 CACHE STRING "Project patch version number.")
mark_as_advanced(VERSION_MAJOR VERSION_MINOR VERSION_PATCH)

# Compile mode, set it to be "true" if CUDA is available!!
set(COMPILE_MODE_CUDA true)

if(NOT ${COMPILE_MODE_CUDA})
add_definitions(-DCPU_ONLY)
endif()

find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})

#find_package(CBLAS REQUIRED)
#include_directories(${BLAS_INCLUDE_DIRS})

#find_package(Qt5Widgets REQUIRED)
#find_package(Qt5UiTools REQUIRED)

if(${COMPILE_MODE_CUDA})
find_package(CUDA REQUIRED)
list(APPEND CUDA_NVCC_FLAGS "-std=c++11;-O2;-DVERBOSE")
include_directories(${CUDA_INCLUDE_DIRS})
endif()

#qt5_wrap_cpp(GSBN_VIEW_MOC include/gsbn_view/Ui.hpp)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(include)
if(${COMPILE_MODE_CUDA})
file(GLOB SOURCES_GSBN "src/gsbn/*.cpp" "src/gsbn/*/*.cpp" "src/gsbn/*/*/*.cpp" "src/gsbn/*.cu" "src/gsbn/*/*.cu" "src/gsbn/*/*/*.cu")
else()
file(GLOB SOURCES_GSBN "src/gsbn/*.cpp" "src/gsbn/*/*.cpp" "src/gsbn/*/*/*.cpp")
endif()
file(GLOB SOURCES_GSBN_SIM "src/gsbn_sim/*.cpp")
file(GLOB SOURCES_GSBN_VIEW "src/gsbn_view/*.cpp")
file(GLOB SOURCES_GSBN_BIN2TXT "src/gsbn_bin2txt/*.cpp")

set(CMAKE_CXX_FLAGS "-std=c++11 -w -pedantic -g -O2")
set(PROJECT_LINK_LIBS)

PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS src/gsbn/proto/gsbn.proto)

if(${COMPILE_MODE_CUDA})
cuda_add_library(gsbn SHARED ${SOURCES_GSBN} ${PROTO_SRCS} ${PROTO_HDRS})
cuda_add_executable(gsbn_sim ${SOURCES_GSBN_SIM})
target_link_libraries(gsbn_sim ${PROJECT_LINK_LIBS} ${PROTOBUF_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_curand_LIBRARY} gsbn)
else()
add_library(gsbn SHARED ${SOURCES_GSBN} ${PROTO_SRCS} ${PROTO_HDRS})
add_executable(gsbn_sim ${SOURCES_GSBN_SIM})
target_link_libraries(gsbn_sim ${PROJECT_LINK_LIBS} ${PROTOBUF_LIBRARIES} gsbn)
endif()




#add_executable(gsbn_sim ${SOURCES_GSBN_SIM})
#add_dependencies(gsbn_sim gsbn)
#target_link_libraries(gsbn_sim ${PROJECT_LINK_LIBS} ${PROTOBUF_LIBRARIES} ${BLAS_LIBRARIES} gsbn)

#add_executable(gsbn_view ${SOURCES_GSBN_VIEW} ${GSBN_VIEW_MOC})
#add_dependencies(gsbn_view gsbn)
#target_link_libraries(gsbn_view ${PROJECT_LINK_LIBS} ${PROTOBUF_LIBRARY} ${BLAS_LIBRARIES} Qt5::Widgets Qt5::UiTools gsbn)

#add_executable(gsbn_bin2txt ${SOURCES_GSBN_BIN2TXT})
#add_dependencies(gsbn_bin2txt gsbn)
#target_link_libraries(gsbn_bin2txt ${PROJECT_LINK_LIBS} ${PROTOBUF_LIBRARIES} ${BLAS_LIBRARIES} gsbn)

#add_executable(test_Mcu test/ghcu/Mcu.cpp)
#target_link_libraries(test_Mcu ${PROJECT_LINK_LIBS})

#enable_testing()
#add_test(test_Mcu test_Mcu)

# add a target to generate API documentation with Doxygen
find_package(Doxygen)
option(BUILD_DOCUMENTATION "Create and install the HTML based API documentation (requires Doxygen)" ${DOXYGEN_FOUND})

if(BUILD_DOCUMENTATION)
    if(NOT DOXYGEN_FOUND)
        message(FATAL_ERROR "Doxygen is needed to build the documentation.")
    endif()

    set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    configure_file(${doxyfile_in} ${doxyfile} @ONLY)

    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)

    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION share/doc)
endif()
