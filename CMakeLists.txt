cmake_minimum_required(VERSION 2.8.9)
project(gsbn)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)
# Show verbose command while compiling
#set(CMAKE_VERBOSE_MAKEFILE ON)

# The project version number.
set(VERSION_MAJOR 1 CACHE STRING "Project major version number.")
set(VERSION_MINOR 14 CACHE STRING "Project minor version number.")
set(VERSION_PATCH 0 CACHE STRING "Project patch version number.")
mark_as_advanced(VERSION_MAJOR VERSION_MINOR VERSION_PATCH)

# MODIFY HERE
# Compile mode, set it to be "true" if CUDA is available!!
set(COMPILE_MODE_CUDA false)

if(NOT ${COMPILE_MODE_CUDA})
add_definitions(-DCPU_ONLY)
endif()

# MODIFY HERE
# If protobuf directory is not a standard directory, please specify it.
#list(APPEND CMAKE_PREFIX_PATH "<PATH TO PROTOBUF>")
#list(APPEND CMAKE_PREFIX_PATH "/pdc/vol/caffe/git-c6d93da/amd64_co7/protobuf")
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})

if(${COMPILE_MODE_CUDA})
# MODIFY HERE
# If CUDA directory is not a standard directory, please specify it.
#set(CUDA_TOOLKIT_ROOT_DIR <PATH TO CUDA>)
#set(CUDA_TOOLKIT_ROOT_DIR /pdc/vol/cuda/cuda-7.5)
find_package(CUDA REQUIRED)
list(APPEND CUDA_NVCC_FLAGS "-std=c++11;-O2;-DVERBOSE")
include_directories(${CUDA_INCLUDE_DIRS})
endif()

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(include)
if(${COMPILE_MODE_CUDA})
file(GLOB SOURCES_GSBN "src/gsbn/*.cpp" "src/gsbn/*/*.cpp" "src/gsbn/*/*/*.cpp" "src/gsbn/*.cu" "src/gsbn/*/*.cu" "src/gsbn/*/*/*.cu")
else()
file(GLOB SOURCES_GSBN "src/gsbn/*.cpp" "src/gsbn/*/*.cpp" "src/gsbn/*/*/*.cpp")
endif()
file(GLOB SOURCES_GSBN_SIM "src/gsbn_sim/*.cpp")
file(GLOB SOURCES_GSBN_TEST_POWER "test/power/*.cpp")

set(CMAKE_CXX_FLAGS "-std=c++11 -w -pedantic -g -O2")
set(PROJECT_LINK_LIBS)

# generate cpp for protobuf
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS src/gsbn/proto/gsbn.proto)
# generate python for protobuf, only works when cmake version higher than 3.4
#PROTOBUF_GENERATE_PYTHON(PROTO_PY src/gsbn/proto/gsbn.proto)

if(${COMPILE_MODE_CUDA})
#cuda_add_library(gsbn SHARED ${SOURCES_GSBN} ${PROTO_SRCS} ${PROTO_HDRS} ${PROTO_PY})
cuda_add_library(gsbn SHARED ${SOURCES_GSBN} ${PROTO_SRCS} ${PROTO_HDRS})
cuda_add_executable(gsbn_sim ${SOURCES_GSBN_SIM})
target_link_libraries(gsbn_sim gsbn ${PROJECT_LINK_LIBS} ${PROTOBUF_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_curand_LIBRARY})
else()
add_library(gsbn SHARED ${SOURCES_GSBN} ${PROTO_SRCS} ${PROTO_HDRS})
add_executable(gsbn_sim ${SOURCES_GSBN_SIM})
target_link_libraries(gsbn_sim gsbn ${PROJECT_LINK_LIBS} ${PROTOBUF_LIBRARIES})
endif()

# test program
add_executable(gsbn_test_power ${SOURCES_GSBN_TEST_POWER})
target_link_libraries(gsbn_test_power ${PROJECT_LINK_LIBS})

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
