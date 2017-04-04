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
set(VERSION_MINOR 15 CACHE STRING "Project minor version number.")
set(VERSION_PATCH 1 CACHE STRING "Project patch version number.")
mark_as_advanced(VERSION_MAJOR VERSION_MINOR VERSION_PATCH)


# Custom functions

# gsbn_protobuf_generate_py:
# Generate python script for protobuf file. Since CMake with version lower than
#   3.4 is not able to generate python protobuf outputs.
function(gsbn_protobuf_generate_cpp_py output_dir srcs_var hdrs_var python_var)
	if(NOT ARGN)
		message(SEND_ERROR "Error: gsbn_protobuf_generate_cpp_py() called without any proto files")
		return()
	endif()
	
	if(PROTOBUF_GENERATE_CPP_APPEND_PATH)
		# Create an include path for each file specified
		foreach(fil ${ARGN})
			get_filename_component(abs_fil ${fil} ABSOLUTE)
			get_filename_component(abs_path ${abs_fil} PATH)
			list(FIND _protoc_include ${abs_path} _contains_already)
			if(${_contains_already} EQUAL -1)
				list(APPEND _protoc_include -I ${abs_path})
			endif()
		endforeach()
	else()
		set(_protoc_include -I ${CMAKE_CURRENT_SOURCE_DIR})
	endif()
	
	if(DEFINED PROTOBUF_IMPORT_DIRS)
		foreach(dir ${PROTOBUF_IMPORT_DIRS})
			get_filename_component(abs_path ${dir} ABSOLUTE)
			list(FIND _protoc_include ${abs_path} _contains_already)
			if(${_contains_already} EQUAL -1)
				list(APPEND _protoc_include -I ${abs_path})
			endif()
		endforeach()
	endif()
	
	set(${srcs_var})
	set(${hdrs_var})
	set(${python_var})
	foreach(fil ${ARGN})
		get_filename_component(abs_fil ${fil} ABSOLUTE)
		get_filename_component(fil_we ${fil} NAME_WE)
		
		list(APPEND ${srcs_var} "${output_dir}/${fil_we}.pb.cc")
		list(APPEND ${hdrs_var} "${output_dir}/${fil_we}.pb.h")
		list(APPEND ${python_var} "${output_dir}/${fil_we}_pb2.py")
		
		add_custom_command(
			OUTPUT "${output_dir}/${fil_we}.pb.cc"
							"${output_dir}/${fil_we}.pb.h"
							"${output_dir}/${fil_we}_pb2.py"
			COMMAND ${CMAKE_COMMAND} -E make_directory "${output_dir}"
			COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --cpp_out    ${output_dir} ${_protoc_include} ${abs_fil}
			COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} --python_out ${output_dir} ${_protoc_include} ${abs_fil}
			DEPENDS ${abs_fil}
			COMMENT "Running C++/Python protocol buffer compiler on ${fil}" VERBATIM )
	endforeach()
	
	set_source_files_properties(${${srcs_var}} ${${hdrs_var}} ${${python_var}} PROPERTIES GENERATED TRUE)
	set(${srcs_var} ${${srcs_var}} PARENT_SCOPE)
	set(${hdrs_var} ${${hdrs_var}} PARENT_SCOPE)
	set(${python_var} ${${python_var}} PARENT_SCOPE)
endfunction()

# MODIFY HERE
# Compile mode, set it to be "true" if CUDA is available!!
set(COMPILE_MODE_CUDA false)
# Compile mode, set it to be "true" if MPI is available!!
set(COMPILE_MODE_MPI true)

if(NOT ${COMPILE_MODE_CUDA})
add_definitions(-DCPU_ONLY)
endif()
if(NOT ${COMPILE_MODE_MPI})
add_definitions(-DNO_MPI)
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

if(${COMPILE_MODE_MPI})
# MODIFY HERE
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
set(MPI_CXX_COMPILER mpicxx.mpich2)
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
file(GLOB PROTO_GSBN "src/gsbn/proto/*.proto")

set(CMAKE_CXX_FLAGS "-std=c++11 -w -pedantic -g -O2")
set(PROJECT_LINK_LIBS)

# generate cpp for protobuf
gsbn_protobuf_generate_cpp_py(${CMAKE_CURRENT_BINARY_DIR} PROTO_SRCS PROTO_HDRS PROTO_PY ${PROTO_GSBN})

if(${COMPILE_MODE_CUDA})
	cuda_add_library(gsbn SHARED ${SOURCES_GSBN} ${PROTO_SRCS} ${PROTO_HDRS} ${PROTO_PY} )
	cuda_add_executable(gsbn_sim ${SOURCES_GSBN_SIM})
	target_link_libraries(gsbn_sim gsbn ${PROJECT_LINK_LIBS} ${PROTOBUF_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} ${CUDA_curand_LIBRARY})
	if(${COMPILE_MODE_MPI})
		target_link_libraries(gsbn_sim ${MPI_LIBRARIES})
		if(MPI_COMPILE_FLAGS)
			set_target_properties(gsbn_sim PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
		endif()
		if(MPI_LINK_FLAGS)
				set_target_properties(hello PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
		endif()
	endif()
else()
	add_library(gsbn SHARED ${SOURCES_GSBN} ${PROTO_SRCS} ${PROTO_HDRS} ${PROTO_PY} )
	add_executable(gsbn_sim ${SOURCES_GSBN_SIM})
	target_link_libraries(gsbn_sim gsbn ${PROJECT_LINK_LIBS} ${PROTOBUF_LIBRARIES})
	if(${COMPILE_MODE_MPI})
		target_link_libraries(gsbn_sim ${MPI_LIBRARIES})
		if(MPI_COMPILE_FLAGS)
			set_target_properties(gsbn_sim PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
		endif()
		if(MPI_LINK_FLAGS)
				set_target_properties(hello PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
		endif()
	endif()
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
